"""
saepipe/sae.py — where SAE encoder weights come from.

Interface [generic]: an encoder is a dict on one device
    {"W_enc": [d_in, d_sae], "b_enc": [d_sae], "threshold": [d_sae]}      (JumpReLU; threshold 0 = ReLU)
Any SAE family with that encoder form can be used by encode.py / pipeline.py. TopK-type SAEs are NOT this
form (selection per token, not per feature) and need a different kernel.

GemmaScope2 [specific adapter]: file layout of google/gemma-scope-2-{size}-{pt|it} on the HF hub
(params.safetensors with w_enc, b_enc, threshold, w_dec; "<hook>_all" folders for the all-layer widths).
A new SAE release = a new adapter class with .load(layer, device, dtype, mode) (and optionally .convert_bf16).
"""
import os
import torch
from safetensors import safe_open

ALL_LAYER_WIDTHS = {"16k", "262k"}      # Gemma Scope 2: every layer, folder "<hook>_all"
SUBSET_WIDTHS = {"65k", "1m"}           # Gemma Scope 2: selected layers, folder "<hook>"


class GemmaScope2:
    def __init__(self, model_size, hook="resid_post", width="262k", l0="small", variant="pt", bf16_dir=None):
        self.model_size, self.hook, self.width, self.l0, self.variant = model_size, hook, width, l0, variant
        self.bf16_dir = bf16_dir
        self.repo = f"google/gemma-scope-2-{model_size}-{variant}"

    def file(self, layer):
        folder = f"{self.hook}_all" if self.width in ALL_LAYER_WIDTHS else self.hook
        return f"{folder}/layer_{layer}_width_{self.width}_l0_{self.l0}/params.safetensors"

    def path(self, layer):
        from huggingface_hub import hf_hub_download
        return hf_hub_download(self.repo, self.file(layer), repo_type="model")

    def bf16_path(self, layer):
        return None if self.bf16_dir is None else f"{self.bf16_dir}/{self.model_size}/layer_{layer}.safetensors"

    def load(self, layer, device, dtype=torch.bfloat16, mode="hf_cpu"):
        """Encoder only (w_enc / b_enc / threshold), never w_dec.
        mode hf_cpu: fp32 HF file -> CPU cast -> device;  hf_gpu: fp32 HF file -> device -> cast there;
             bf16:   pre-converted bf16 encoder file (convert_bf16) -> device (falls back to hf_gpu if absent)."""
        bp = self.bf16_path(layer)
        if mode == "bf16" and (bp is None or not os.path.exists(bp)):
            mode = "hf_gpu"
        if mode == "bf16":
            with safe_open(bp, framework="pt", device=str(device)) as f:
                return {"W_enc": f.get_tensor("w_enc").to(dtype), "b_enc": f.get_tensor("b_enc").to(dtype),
                        "threshold": f.get_tensor("threshold").to(dtype)}
        out = {}
        src = "cpu" if mode == "hf_cpu" else str(device)
        with safe_open(self.path(layer), framework="pt", device=src) as f:
            keys = set(f.keys())
            out["W_enc"] = f.get_tensor("w_enc").to(dtype).to(device, non_blocking=True)
            d_sae = out["W_enc"].shape[1]
            out["b_enc"] = (f.get_tensor("b_enc") if "b_enc" in keys else torch.zeros(d_sae)).to(dtype).to(device)
            out["threshold"] = f.get_tensor("threshold").to(dtype).to(device)
        return out

    def threshold(self, layer):
        with safe_open(self.path(layer), framework="pt", device="cpu") as f:
            return f.get_tensor("threshold").float().numpy()

    def convert_bf16(self, layer):
        """Write the encoder-only bf16 copy (2.8 GB instead of 11.3 GB for 27b 262k). Returns path."""
        from safetensors.torch import save_file
        dst = self.bf16_path(layer)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst):
            with safe_open(self.path(layer), framework="pt", device="cpu") as f:
                t = {k: f.get_tensor(k).to(torch.bfloat16).contiguous() for k in ("w_enc", "b_enc", "threshold")}
            save_file(t, dst + ".tmp")
            os.replace(dst + ".tmp", dst)
        return dst


def load_parallel(source, layers, devs, dtype=torch.bfloat16, mode="hf_cpu", parallel=True):
    """Encoders for `layers`, each on devs[L]; parallel=True: one loader thread per GPU. [generic]"""
    if not parallel:
        return {L: source.load(L, devs[L], dtype=dtype, mode=mode) for L in layers}
    from concurrent.futures import ThreadPoolExecutor
    by_dev = {}
    for L in layers:
        by_dev.setdefault(devs[L], []).append(L)
    out = {}

    def work(d, Ls):
        with torch.cuda.device(d):
            for L in Ls:
                out[L] = source.load(L, d, dtype=dtype, mode=mode)
        torch.cuda.synchronize(d)

    with ThreadPoolExecutor(max_workers=len(by_dev)) as ex:
        for f in [ex.submit(work, d, Ls) for d, Ls in by_dev.items()]:
            f.result()
    return out
