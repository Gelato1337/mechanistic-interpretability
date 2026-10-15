#!/usr/bin/env python3
"""
bench_saelens.py — the community-standard baseline the capture engine is measured against: SAELens on a
TransformerLens model, as the SAELens tutorials use it.

    python benchmarks/bench_saelens.py --model 27b --prompts thesis/data/prompts.csv \
        --domains par_lang_english,disjoint_lang_finnish,disjoint_lang_german,div_code_python --n-per-domain 32 \
        --out runs/bench_saelens
    python scripts/capture.py --model 27b --prompts thesis/data/prompts.csv --domains <same> --n-per-domain 32 \
        --skip-bos 0 --out runs/bench_saepipe
    python scripts/compare_captures.py runs/bench_saelens runs/bench_saepipe

What it does, per layer L: load the Gemma Scope 2 SAE with SAE.from_pretrained (SAELens defaults: float32 encoder and
decoder), then for every batch run_with_cache(names_filter=blocks.L.hook_resid_post, stop_at_layer=L+1) and
sae.encode on the residual of the real tokens; the nonzero entries are the output. <bos> is encoded (as in the
measured comparison), so use --skip-bos 0 on the engine side. Same tokenizer, same token ids and the same output
format as scripts/capture.py (tokens.npz + layer_<L>/part_*.npz + run.json with timings).

Two TransformerLens issues had to be worked around for Gemma 3 27B (TransformerLens 2.17):
  * blocks are mapped to GPUs as i // (n_layers // n_devices); 62 layers on 8 GPUs map the last blocks past the last GPU,
    so n_devices is the largest GPU count that divides the layer count (27B: 2);
  * its Gemma 3 config lacks query_pre_attn_scalar and the linear RoPE scaling of the global layers, so its residual
    stream differs from Hugging Face's by several per cent in the deep layers; --tl-fix 1 (default) patches both
    (tl_gemma3_fix.py) and brings it to the bf16 noise floor.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from saepipe.io import ShardWriter
from capture import MODEL_REPOS, load_prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="27b", choices=list(MODEL_REPOS))
    ap.add_argument("--prompts", nargs="+", required=True)
    ap.add_argument("--domains", default=None)
    ap.add_argument("--n-per-domain", type=int, default=None)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--tl-fix", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from sae_lens import SAE, HookedSAETransformer
    from transformers import AutoConfig, AutoTokenizer
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    repo = MODEL_REPOS[a.model]; tok = AutoTokenizer.from_pretrained(repo); tok.padding_side = "right"
    prompts = load_prompts(a.prompts, a.domains.split(",") if a.domains else None, a.n_per_domain)
    ids = [tok(t, truncation=True, max_length=a.max_seq_len)["input_ids"] for _, _, t in prompts]
    ntok = np.array([len(i) for i in ids], np.int64); row_off = np.concatenate([[0], np.cumsum(ntok)[:-1]])
    np.savez(out / "tokens.npz", prompt_ids=np.array([p[0] for p in prompts], np.int64), domains=np.array([p[1] for p in prompts]),
             n_tokens=ntok, token_ids=np.concatenate([np.asarray(i, np.int32) for i in ids]))

    hf_cfg = AutoConfig.from_pretrained(repo); text_cfg = getattr(hf_cfg, "text_config", hf_cfg)
    n_gpu = max(1, torch.cuda.device_count()); n_dev = max(d for d in range(1, n_gpu + 1) if text_cfg.num_hidden_layers % d == 0)
    t0 = time.perf_counter()
    model = HookedSAETransformer.from_pretrained_no_processing(repo, dtype=torch.bfloat16, n_devices=n_dev, device="cuda").eval()
    if a.tl_fix:
        from tl_gemma3_fix import fix_tl_gemma3
        fix_tl_gemma3(model, text_cfg)
    t_model = time.perf_counter() - t0
    n_layers = model.cfg.n_layers; layers = list(range(n_layers)) if a.layers == "all" else [int(x) for x in a.layers.split(",")]
    dev_in = next(model.embed.parameters()).device
    enc_batch = lambda b: tok([p[2] for p in b], return_tensors="pt", padding=True, truncation=True, max_length=a.max_seq_len)
    with torch.no_grad():   # untimed warmup
        e = enc_batch(prompts[:a.batch_size]); model(e["input_ids"].to(dev_in), attention_mask=e["attention_mask"].to(dev_in), return_type=None)
    torch.cuda.synchronize()
    writer = ShardWriter(out, n_threads=2); t_sae = 0.0; t0 = time.perf_counter()
    for L in layers:
        dev = next(model.blocks[L].parameters()).device; ts = time.perf_counter()
        sae = SAE.from_pretrained(f"gemma-scope-2-{a.model}-pt-res-all", f"layer_{L}_width_262k_l0_small", device=str(dev))
        sae = sae[0] if isinstance(sae, tuple) else sae; t_sae += time.perf_counter() - ts
        hook = f"blocks.{L}.hook_resid_post"
        for i in range(0, len(prompts), a.batch_size):
            e = enc_batch(prompts[i:i + a.batch_size]); am = e["attention_mask"]
            with torch.no_grad():
                _, cache = model.run_with_cache(e["input_ids"].to(dev_in), attention_mask=am.to(dev_in), names_filter=hook,
                                                stop_at_layer=L + 1, return_type=None)
                resid = cache[hook]; x = resid[am.bool().to(resid.device)]
                feats = sae.encode(x.to(device=sae.W_enc.device, dtype=sae.W_enc.dtype))
                nz = feats.nonzero(); vals = feats[nz[:, 0], nz[:, 1]].float().cpu().numpy(); nz = nz.cpu().numpy()
            # real-token rows of this batch, in prompt order
            rows = np.concatenate([row_off[k] + np.arange(int(am[k - i].sum())) for k in range(i, min(i + a.batch_size, len(prompts)))])
            writer.put(L, rows[nz[:, 0]].astype(np.int32), nz[:, 1].astype(np.int32), vals.astype(np.float32))
            del cache, resid, x, feats
        del sae; torch.cuda.empty_cache(); print(f"layer {L} done", flush=True)
    writer.close(); t_compute = time.perf_counter() - t0
    (out / "run.json").write_text(json.dumps(dict(args=vars(a), engine="saelens", model_repo=repo, tl_n_devices=n_dev, n_layers=n_layers,
        layers=layers, n_prompts=len(prompts), n_tokens=int(ntok.sum()), firings_per_layer={str(k): int(v) for k, v in writer.nnz.items()},
        t_load_model_s=t_model, t_load_sae_s=t_sae, t_compute_s=t_compute, t_compute_excl_sae_load_s=t_compute - t_sae), indent=2))
    print(f"compute {t_compute:.1f} s, of which SAE load {t_sae:.1f} s", flush=True)


if __name__ == "__main__":
    main()
