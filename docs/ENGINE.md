# The capture engine (`saepipe`)

`scripts/capture.py` runs a Gemma 3 model over a prompt set and encodes the residual stream after every
requested layer with the Gemma Scope 2 JumpReLU SAEs. The work is done by the `saepipe` package.

## What it does differently from a plain PyTorch loop

| step | plain approach | saepipe |
|---|---|---|
| forward pass | one full forward per layer of interest | one forward per micro-batch, all layers at once |
| SAE weights | whole file (encoder + decoder, fp32) per layer | encoder only, resident on the GPU that holds its layer |
| encode | dense `x @ W_enc`, threshold, `nonzero`, host sync | one fused Triton kernel: GEMM + bias + threshold + sparse compaction, no dense activation matrix, no host sync |
| GPUs | layers split over GPUs, used one after another | forward-only pipelining: every GPU works on a different micro-batch at every step |
| output | written after the run | streamed by a background writer while the GPUs compute |

## Measured speed and agreement

On Gemma 3 27B, 62 layers, 128 prompts, one node of 8 MI250X dies: 34 s against 1,454 s for a naive per-layer loop
(43×) and 1,038 s for SAELens. The full comparison, the agreement checks and the bf16 threshold effect are in
`BENCHMARKS.md`.

## Validating on new hardware

The kernel source is vendor-neutral Triton; its launch configuration is not. `saepipe/kernels/configs.py`
holds a configuration tuned and validated on MI250X. On any other GPU (NVIDIA Ampere or newer, other AMD
generations) the kernel autotunes over a candidate list, and before trusting it:

1. `python scripts/test_kernel.py --model 27b --layers 0,31` (one GPU): every correctness entry must show
   `only_ref == only_new == 0` against the torch reference of the same precision.
2. A small capture with both encoders, compared:
   ```
   python scripts/capture.py --model 1b --prompts prompts.csv --n-per-domain 4 --out runs/check_triton
   python scripts/capture.py --model 1b --prompts prompts.csv --n-per-domain 4 --encoder torch --out runs/check_torch
   python scripts/compare_captures.py runs/check_triton runs/check_torch
   ```

The NVIDIA path has not been run. Configurations observed to miscompile on MI250X are listed in
`configs.KNOWN_BAD` and refused.

## Module map

| module | role | scope |
|---|---|---|
| `saepipe/backend.py` | GPU vendor detection, bf16 capability check | generic |
| `saepipe/kernels/jumprelu.py` | fused JumpReLU encode kernel (Triton) | generic source |
| `saepipe/kernels/configs.py` | launch configurations per vendor | MI250X tuned; others autotune |
| `saepipe/sae.py` | encoder loading; `GemmaScope2` knows the Hugging Face repo layout | interface generic, adapter Gemma Scope 2 |
| `saepipe/model.py` | model loading, layer-to-GPU map, pipeline stages, exact layer-0 inputs | Hugging Face decoders with `.layers`; tested on Gemma 3 |
| `saepipe/encode.py` | `TritonJumpReLU` (async), `TorchJumpReLU` (reference) | generic |
| `saepipe/pipeline.py` | forward-only pipeline scheduler | residual stream (`resid_post`) only |
| `saepipe/io.py` | streaming shard writer and reader | generic |

Not supported: MLP or attention-output hook points, transcoders, TopK SAEs.
