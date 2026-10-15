#!/usr/bin/env python3
"""
capture.py — run a Gemma 3 model over a prompt set and encode the residual stream of every requested layer
with the Gemma Scope 2 JumpReLU SAEs, writing the sparse codes to disk.

    python scripts/capture.py --model 27b --prompts prompts.csv --out runs/27b_all
    python scripts/capture.py --model 1b  --prompts prompts.csv --domains par_lang_english,par_lang_finnish \
                              --layers 0,6,12 --out runs/1b_small

Engine: saepipe (one forward pass per micro-batch for all layers, a fused Triton JumpReLU encode on every GPU,
forward-only pipelining across GPUs, a streaming writer). See docs/ENGINE.md.

Prompt files: CSV with columns prompt_id, micro_domain, text (e.g. the output of thesis/data/build_dataset.py). Prompts
are taken per domain in prompt_id order (--n-per-domain caps each domain); domains in sorted order unless
--domains gives an explicit list.

Output directory:
    tokens.npz          prompt_ids, domains, n_tokens (tokens per prompt incl. <bos>), token_ids (all rows)
    layer_<L>/part_*.npz rows (int32 global token row), feats (int32 SAE latent), vals (float32 activation)
    run.json            arguments, environment, timings, firings per layer
Row r belongs to prompt p where offsets[p] <= r < offsets[p+1], offsets = cumsum(n_tokens) with a leading 0;
position 0 of every prompt is <bos>, which is not encoded with --skip-bos 1 (the default, as in the thesis).

Activation value: the Gemma Scope / SAELens definition z = pre * 1[pre > theta] (--act gemma, default). The
fp32 epilogue (default) compares pre with theta in fp32, so no firing just above theta is lost to bf16
rounding; --bf16-epilogue reproduces the rounding of a plain bf16 encode.
"""
import argparse, json, os, platform, socket, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from saepipe.backend import vendor
from saepipe.encode import TritonJumpReLU, TorchJumpReLU
from saepipe.io import ShardWriter
from saepipe.kernels.configs import TUNED
from saepipe.model import load_hf_decoder, layer_device
from saepipe.pipeline import Pipeline
from saepipe.sae import GemmaScope2, load_parallel

MODEL_REPOS = {"270m": "google/gemma-3-270m", "1b": "google/gemma-3-1b-pt", "4b": "google/gemma-3-4b-pt",
               "12b": "google/gemma-3-12b-pt", "27b": "google/gemma-3-27b-pt"}
MODEL_GB_BF16 = {"270m": 0.6, "1b": 2.2, "4b": 8.5, "12b": 24.5, "27b": 54.0}


def load_prompts(paths, domains=None, n_per_domain=None):
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    doms = domains or sorted(df["micro_domain"].unique())
    out = []
    for d in doms:
        sub = df[df["micro_domain"] == d].sort_values("prompt_id")
        if n_per_domain:
            sub = sub.head(n_per_domain)
        if len(sub) == 0:
            raise SystemExit(f"domain {d!r} not found in {paths}")
        out += [(int(r.prompt_id), d, r.text) for r in sub.itertuples()]
    return out


def batches(n, ntok, budget, max_bs):
    """Contiguous micro-batches whose padded size (prompts x longest prompt) stays within `budget` tokens."""
    i = 0
    while i < n:
        j, mx = i, 0
        while j < n and j - i < max_bs:
            m2 = max(mx, int(ntok[j]))
            if j > i and m2 * (j - i + 1) > budget:
                break
            mx, j = m2, j + 1
        yield i, j
        i = j


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(MODEL_REPOS))
    ap.add_argument("--prompts", required=True, nargs="+", help="one or more prompt CSV files")
    ap.add_argument("--domains", default=None, help="comma-separated micro_domain list (default: all, sorted)")
    ap.add_argument("--n-per-domain", type=int, default=None)
    ap.add_argument("--layers", default="all", help="'all' or comma-separated layer indices")
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", default="262k", choices=["16k", "262k"], help="Gemma Scope 2 all-layer width")
    ap.add_argument("--l0", default="small")
    ap.add_argument("--sae-bf16-dir", default=None, help="encoder-only bf16 copies from convert_saes.py (faster load)")
    ap.add_argument("--act", default="gemma", choices=["gemma", "shifted"], help="gemma: pre if pre>theta; shifted: pre-theta")
    ap.add_argument("--bf16-epilogue", action="store_true", help="compare pre with theta in bf16 (plain bf16 encode)")
    ap.add_argument("--encoder", default="triton", choices=["triton", "torch"], help="torch = slow reference")
    ap.add_argument("--skip-bos", type=int, default=1)
    ap.add_argument("--max-seq-len", type=int, default=512, help="truncation (the thesis used 512)")
    ap.add_argument("--token-budget", type=int, default=8192, help="padded tokens per micro-batch")
    ap.add_argument("--max-batch", type=int, default=64)
    ap.add_argument("--wave", type=int, default=64, help="micro-batches per readback wave")
    ap.add_argument("--cap-per-token", type=int, default=1024, help="output slots per token (overflow re-encodes exactly)")
    ap.add_argument("--writer-threads", type=int, default=4)
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_REPOS[a.model]); tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompts = load_prompts(a.prompts, a.domains.split(",") if a.domains else None, a.n_per_domain)
    ids = [tok(t, truncation=True, max_length=a.max_seq_len)["input_ids"] for _, _, t in prompts]
    ntok = np.array([len(i) for i in ids], np.int64); row_off = np.concatenate([[0], np.cumsum(ntok)[:-1]])
    np.savez(out / "tokens.npz", prompt_ids=np.array([p[0] for p in prompts], np.int64), domains=np.array([p[1] for p in prompts]),
             n_tokens=ntok, token_ids=np.concatenate([np.asarray(i, np.int32) for i in ids]))
    print(f"{len(prompts)} prompts, {int(ntok.sum()):,} tokens", flush=True)

    t0 = time.perf_counter()
    model, tm = load_hf_decoder(MODEL_REPOS[a.model], model_gb=MODEL_GB_BF16[a.model], spread=True)
    t_model = time.perf_counter() - t0
    n_layers = len(tm.layers)
    layers = list(range(n_layers)) if a.layers == "all" else [int(x) for x in a.layers.split(",")]
    src = GemmaScope2(a.model, hook="resid_post", width=a.width, l0=a.l0, bf16_dir=a.sae_bf16_dir)
    t0 = time.perf_counter()
    saes = load_parallel(src, layers, {L: layer_device(tm, L) for L in layers}, mode="bf16" if a.sae_bf16_dir else "hf_gpu")
    t_sae = time.perf_counter() - t0
    if a.encoder == "triton":
        enc = TritonJumpReLU(saes, act=a.act, fp32_epilogue=not a.bf16_epilogue, config="tuned" if TUNED.get(vendor()) else "autotune",
                             cap_per_token=a.cap_per_token)
        enc.warmup()
    else:
        enc = TorchJumpReLU(saes, act=a.act)
    pipe = Pipeline(tm, layers, enc, skip_bos=bool(a.skip_bos))
    writer = ShardWriter(out, n_threads=a.writer_threads)

    t0 = time.perf_counter()
    mbs = list(batches(len(prompts), ntok, a.token_budget, a.max_batch))
    emit = lambda L, r, f, v: writer.put(L, r.astype(np.int32), f.astype(np.int32), v.astype(np.float32))
    try:
        for w0 in range(0, len(mbs), a.wave):
            preps = {}
            for m, (i, j) in enumerate(mbs[w0:w0 + a.wave]):
                e = tok([p[2] for p in prompts[i:j]], return_tensors="pt", padding=True, truncation=True, max_length=a.max_seq_len)
                rows = [row_off[k] + np.arange(int(e["attention_mask"][k - i].sum())) for k in range(i, j)]
                preps[m] = pipe.prepare(e, rows)
            pipe.readback(pipe.enqueue(preps), preps, emit)
            print(f"  micro-batches {min(w0 + a.wave, len(mbs))}/{len(mbs)}", flush=True)
    finally:
        pipe.close(); writer.close()
    t_compute = time.perf_counter() - t0

    info = dict(args=vars(a), model_repo=MODEL_REPOS[a.model], sae_repo=src.repo, n_layers=n_layers, layers=layers,
                n_prompts=len(prompts), n_tokens=int(ntok.sum()), firings_per_layer={str(k): int(v) for k, v in writer.nnz.items()},
                overflow_reruns=getattr(enc, "overflow_reruns", 0), t_load_model_s=t_model, t_load_sae_s=t_sae, t_compute_s=t_compute,
                env=dict(host=socket.gethostname(), python=platform.python_version(), torch=torch.__version__,
                         hip=getattr(torch.version, "hip", None), cuda=torch.version.cuda, n_gpus=torch.cuda.device_count(), vendor=vendor()))
    (out / "run.json").write_text(json.dumps(info, indent=2))
    print(f"done: {sum(writer.nnz.values()):,} firings, compute {t_compute:.1f} s", flush=True)


if __name__ == "__main__":
    main()
