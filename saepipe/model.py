"""
saepipe/model.py — the language model side. [generic for Hugging Face decoder LMs whose decoder layers live in
`.layers` (Gemma, Llama, Mistral, Qwen, ...); tested only with Gemma 3 27B]

* load_hf_decoder: bf16 model split over all visible GPUs (accelerate device_map, balanced max_memory)
* build_stages: consecutive decoder layers on the same GPU = one pipeline stage
* Layer0Capture: runs the model's OWN preamble (embedding, causal mask, rotary tables, position ids)
  and captures exactly what decoder layer 0 receives, so driving the layers by hand is numerically
  identical to model.forward.
"""
import torch


class _Stop(Exception):
    pass


def balanced_max_memory(model_gb, n_gpus, cap_gb=38.0, floor_gb=8.0, cpu_gb=40):
    per = min(max(model_gb * 1.2 / n_gpus, floor_gb), cap_gb)
    mm = {i: (f"{per:.0f}GiB" if per >= 1 else f"{int(per * 1024)}MiB") for i in range(n_gpus)}
    mm["cpu"] = f"{cpu_gb}GiB"
    return mm


def even_device_map(repo, n_gpus):
    """Decoder layer i -> GPU i*n/L; every other module (embeddings, norms, rotary, lm_head, vision tower)
    -> GPU 0. Built on empty weights, so it costs no memory. Puts layers (and their SAEs) on every GPU even
    for models that would fit on one."""
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM
    with init_empty_weights():
        m = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(repo))
    tm = text_model(m)
    L = len(tm.layers)
    dm = {}

    def contains(mod, target):
        return any(x is target for x in mod.modules())

    def walk(mod, name):
        if mod is tm:
            for cname, c in mod.named_children():
                full = f"{name}.{cname}" if name else cname
                if c is tm.layers:
                    for i in range(L):
                        dm[f"{full}.{i}"] = i * n_gpus // L
                else:
                    dm[full] = 0
            return
        if contains(mod, tm):
            for cname, c in mod.named_children():
                walk(c, f"{name}.{cname}" if name else cname)
            return
        dm[name] = 0

    walk(m, "")
    return dm


def load_hf_decoder(repo, model_gb=None, dtype=torch.bfloat16, spread=False):
    """(model, decoder) with the decoder's layers spread over all GPUs. spread=True: no 8 GB per-GPU floor,
    so small models are split over every GPU too (more pipeline stages = more GPUs encoding at once)."""
    from transformers import AutoModelForCausalLM
    kw = {"torch_dtype": dtype, "low_cpu_mem_usage": True}
    n = torch.cuda.device_count()
    if n > 1 and spread:
        kw["device_map"] = even_device_map(repo, n)   # layers evenly over all GPUs ("balanced" can put a small model on one GPU -> OOM)
    elif n > 1:
        kw["device_map"] = "auto"
        if model_gb is not None:
            kw["max_memory"] = balanced_max_memory(model_gb, n)
    else:
        kw["device_map"] = "cuda:0"
    model = AutoModelForCausalLM.from_pretrained(repo, **kw).eval()
    return model, text_model(model)


def text_model(model):
    """The decoder module that owns `.layers` (handles multimodal wrappers such as Gemma3ForConditionalGeneration)."""
    tm = model.model if hasattr(model, "model") else model
    if not hasattr(tm, "layers"):
        for attr in ["text_model", "language_model", "decoder"]:
            c = getattr(tm, attr, None)
            if c is not None and hasattr(c, "layers"):
                return c
    return tm


def layer_device(tm, layer):
    return next(tm.layers[layer].parameters()).device


def build_stages(tm, last_layer):
    """[(device, [layer ids])] for layers 0..last_layer, consecutive layers on one device grouped."""
    stages = []
    for i in range(last_layer + 1):
        d = layer_device(tm, i)
        if stages and stages[-1][0] == d:
            stages[-1][1].append(i)
        else:
            stages.append((d, [i]))
    return stages


def to_device(obj, d):
    if torch.is_tensor(obj):
        return obj.to(d, non_blocking=True)
    if isinstance(obj, tuple):
        return tuple(to_device(o, d) for o in obj)
    if isinstance(obj, list):
        return [to_device(o, d) for o in obj]
    if isinstance(obj, dict):
        return {k: to_device(v, d) for k, v in obj.items()}
    return obj


class Layer0Capture:
    """capture(inputs) -> (hidden_states, kwargs) that decoder layer 0 would receive. The hook is inert
    outside capture(), so layer 0 can afterwards be called directly."""
    def __init__(self, tm):
        self.tm, self.active, self.got = tm, False, {}
        self.dev_in = tm.embed_tokens.weight.device
        self.h = tm.layers[0].register_forward_pre_hook(self._hook, with_kwargs=True)

    def _hook(self, mod, args, kwargs):
        if not self.active:
            return None
        self.got["h"], self.got["kw"] = args[0], kwargs
        raise _Stop

    def capture(self, enc):
        self.active = True
        with torch.no_grad():
            try:
                self.tm(**enc.to(self.dev_in), use_cache=False)   # no KV-cache object in the kwargs
            except _Stop:
                pass
            finally:
                self.active = False
        return self.got.pop("h"), self.got.pop("kw")

    def remove(self):
        self.h.remove()
