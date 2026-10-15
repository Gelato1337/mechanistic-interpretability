# Speed and agreement of the capture engine

All measurements: Gemma 3 27B (62 layers), Gemma Scope 2 `resid_post` SAEs, width 262k, `l0_small`, on one LUMI
GPU node (4 AMD MI250X = 8 GPU dies of 64 GB, ROCm 6.2, PyTorch 2.7.1). Times are compute after the model is loaded.

## The three methods compared

| method | how it works |
|---|---|
| **naive per-layer loop** (the engine first used for the thesis data) | For each layer: a full forward pass of the model with a hook on that layer; per batch of 4 prompts a dense bf16 encode (`x @ W_enc + b`, threshold, `nonzero`, copy to the host). The model is split over the GPUs with `device_map="auto"`, so one GPU works at a time. SAE files are loaded whole (encoder and decoder, fp32) and cast on the CPU. |
| **SAELens** (`benchmarks/bench_saelens.py`) | The community-standard path: a TransformerLens `HookedSAETransformer`; per layer `SAE.from_pretrained` (fp32) and per batch `run_with_cache(..., stop_at_layer=L+1)` then `sae.encode`. Forward passes stop at the layer, so the cost is a sum of partial forwards. |
| **saepipe** (`scripts/capture.py`) | One forward pass per micro-batch for all layers; bf16 encoder-only copies of the SAEs, resident on the GPU that holds their layer; a fused Triton kernel (GEMM + bias + threshold + sparse compaction, fp32 epilogue, no dense activation matrix, no host sync); forward-only pipelining so all 8 GPUs work on different micro-batches; a background shard writer. |

## 128 prompts, all 62 layers

The first 32 prompts of four domains (English FLORES sentences, Finnish and German Wikipedia paragraphs, Python from
The Stack): 128 prompts, 20,516 tokens, `<bos>` encoded. The same prompts for every method.

| method | compute, SAE loading excluded | speed-up over naive | SAE loading, 62 layers | firings |
|---|---|---|---|---|
| naive per-layer loop | 1,454 s | 1× | 2,114 s | 591,264,799 |
| SAELens (with the TransformerLens fix below) | 1,038 s | 1.4× | 859 s | 601,082,422 |
| one forward for all layers, batched torch bf16 encode | 89 s | 16× | 101 s | 591,264,808 |
| saepipe: + fused Triton encode + pipelining | **34 s** | **43×** | 169 s | 601,586,012 |

Where the time goes (27B, per token): the forward pass costs about 54 GFLOP over all 62 layers and one 262k encode
2.8 GFLOP, 175 GFLOP for all layers. The naive loop runs the forward once per layer, 62 × 54 = 3,348 GFLOP. A single
forward leaves the encode as about three quarters of the work; the fused kernel and pipelining address that part.

SAE loading: encoder-only bf16 copies (`scripts/convert_saes.py`, a one-off conversion of about 26 s per layer)
are loaded straight to the GPU. Loading the full fp32 files and casting on the CPU was the main cost of the naive loop.

## The full thesis prompt set

| method | prompt set | node-hours | basis |
|---|---|---|---|
| naive per-layer loop | 16,500 prompts, `<bos>` encoded | 60.4 | measured |
| SAELens | 16,500 prompts | about 43 | estimate from the 128-prompt run |
| saepipe | 16,500 prompts | about 1.6 | estimate from a 990-prompt run with the same length mix |
| saepipe | 14,000 prompts of the 28 domains, `<bos>` skipped | 0.72 (43 min) | measured |

Capture of the 14,000 prompts (2.2 M tokens) at every layer, saepipe, one node:

| size | layers | compute | SAE loading | firings |
|---|---|---|---|---|
| 270M | 18 | 160 s | 7 s | 0.82 G |
| 1B | 26 | 246 s | 18 s | 1.14 G |
| 4B | 34 | 492 s | 54 s | 1.47 G |
| 12B | 48 | 1,230 s | 156 s | 5.31 G |
| 27B | 62 | 2,573 s | 156 s | 2.74 G |

At this scale the shard writer is not a bottleneck: on a 1 M-token run it had about 1.1 GB/s of capacity against
56 MB/s of output.

## Agreement

- **Engine against engine.** The single-forward engine with the torch encode and the pipelined engine with the torch
  encode give identical firings at all 62 layers. Between the naive loop and the single-forward engine, 9 of
  591 M firings differ.
- **The fused kernel** reproduces a torch encode of the same precision exactly (`scripts/test_kernel.py`).
- **bf16 threshold loss.** A plain bf16 encode rounds the pre-activation before comparing it with the threshold,
  so firings just above the threshold are lost, all on one side: about 1 % of all firings, up to 5 % at some
  middle layers. The fused kernel compares in fp32 and removes that bias at no cost. The remaining difference to a
  full fp32 encode is symmetric and comes from the bf16 encoder weights. This is why saepipe finds about 1.7 % more
  firings than the naive loop in the table above.
- **Noise floor.** Changing only the batch size moves the per-layer Jaccard index of the firing sets to between
  0.992 and 0.99999. Differences of that size between methods are not errors.
- **TransformerLens and Gemma 3.** TransformerLens 2.17's Gemma 3 configuration omits `query_pre_attn_scalar`
  and the linear RoPE scaling of the global layers, so its residual stream differs from Hugging Face's by several
  per cent in the deep layers. `benchmarks/tl_gemma3_fix.py` patches both and brings it to the bf16 noise floor;
  the SAELens row above uses the fix. TransformerLens also maps 62 blocks onto 8 GPUs past the last device;
  the benchmark uses the largest GPU count that divides the layer count.

## Reproducing the comparison

```
python benchmarks/bench_saelens.py --model 27b --prompts thesis/data/prompts.csv \
    --domains par_lang_english,disjoint_lang_finnish,disjoint_lang_german,div_code_python --n-per-domain 32 --out runs/bench_saelens
python scripts/capture.py --model 27b --prompts thesis/data/prompts.csv \
    --domains par_lang_english,disjoint_lang_finnish,disjoint_lang_german,div_code_python --n-per-domain 32 --skip-bos 0 --out runs/bench_saepipe
python scripts/compare_captures.py runs/bench_saelens runs/bench_saepipe
```

Both write their timings to `run.json`. The Finnish Wikipedia domain of the measured runs came from an earlier
version of the dataset, so the prompts differ slightly from these commands; the per-token costs do not depend on it.
The naive loop is not included in this repository.
