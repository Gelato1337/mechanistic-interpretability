# Mechanistic interpretability with sparse autoencoders on Gemma 3

Tools and results from a master's thesis on sparse-autoencoder (SAE) features of Gemma 3 with the Gemma Scope 2
SAEs. The repository has four parts:

1. **[saepipe](#saepipe-fast-sae-feature-capture)**, an engine that records the SAE features of every token at
   every layer, 43 times faster than a naive PyTorch loop and 30 times faster than SAELens.
2. **[A gallery of topic features](#gallery-of-topic-features)** of Gemma 3 27B, served with GitHub Pages.
3. **[A notebook](#inspecting-a-capture)** that shows how to read and look at a capture.
4. **[The thesis scripts](#the-thesis)**: the dataset, the analyses and the feature-class classifier.

```
saepipe/        the capture engine (Python package)
scripts/        command-line tools: capture, aggregate, convert SAEs, kernel test, compare captures
benchmarks/     the SAELens baseline of the speed comparison
docs/           ENGINE.md (how the engine works), BENCHMARKS.md (speed and agreement)
slurm/          example batch jobs for LUMI (AMD MI250X)
gallery/        the topic-feature gallery: builder and the published site (gallery/site/)
notebooks/      inspect_capture.ipynb
thesis/         data/ (the prompt set) and analysis/ (always_on, classes, sharing, topics)
```

## saepipe: fast SAE feature capture

`saepipe` runs a Gemma 3 model over a set of prompts and stores the sparse code of every token at every requested
layer: which Gemma Scope 2 features fire and how strongly.

### Speed

Gemma 3 27B, all 62 layers, Gemma Scope 2 `resid_post` 262k SAEs, 128 prompts (20,516 tokens), one node of
8 AMD MI250X GPU dies:

| method | compute | speed-up |
|---|---|---|
| naive per-layer PyTorch loop | 1,454 s | 1× |
| SAELens on TransformerLens | 1,038 s | 1.4× |
| one forward pass for all layers, batched torch encode | 89 s | 16× |
| **saepipe**: + fused Triton encode + pipelining over GPUs | **34 s** | **43×** |

A full run of 14,000 prompts through all 62 layers of the 27B model takes 43 minutes on one node; the 270M to 12B
models take 3 to 21 minutes. The firings agree exactly with a torch encode of the same precision, and with the
other methods within the batch-size noise floor. Method details, agreement checks and how to repeat the
comparison are in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

### How it works

- **One forward pass for all layers.** Each micro-batch goes through the model once, and every layer's residual
  stream is encoded on the GPU that computed it. The naive approach runs the model once per layer.
- **A fused JumpReLU kernel.** One Triton kernel does the encoder GEMM, adds the bias, applies the threshold and
  writes only the firing features. There is no dense activation matrix and no host synchronisation. The threshold
  is compared in fp32, which avoids the loss of near-threshold firings that a plain bf16 encode has.
- **Pipelining.** The layers are split over the GPUs and different micro-batches move through them at the same
  time, so all GPUs work at once instead of one after another.
- **Lean loading and output.** Only the encoders are loaded, as bf16 copies, each on the GPU that holds its layer.
  A background writer streams the sparse codes to disk while the GPUs compute.

More in [docs/ENGINE.md](docs/ENGINE.md).

### Quick start

Requirements: Python 3.10+, `pip install -r requirements.txt`, access to the gated Hugging Face repositories
`google/gemma-3-*` and `google/gemma-scope-2-*` (`HF_TOKEN` in the environment), and GPUs with room for the model in
bf16 plus the encoders.

```
# optional, once: encoder-only bf16 copies of the SAEs (faster loading)
python scripts/convert_saes.py --model 27b --n-layers 62 --out sae_bf16

# capture: a CSV with columns prompt_id, micro_domain, text
python scripts/capture.py --model 27b --prompts prompts.csv --out runs/my_run --sae-bf16-dir sae_bf16
python scripts/capture.py --model 1b  --prompts prompts.csv --layers 0,6,12 --n-per-domain 50 --out runs/small
```

Output, one directory per run:

| file | contents |
|---|---|
| `tokens.npz` | prompt ids, domains, tokens per prompt, token ids |
| `layer_<L>/part_*.npz` | `rows` (global token index), `feats` (feature id), `vals` (activation), one entry per firing |
| `run.json` | arguments, environment, timings, firings per layer |

`scripts/aggregate.py` reduces a large capture to per-domain statistics per layer.

### Hardware

Developed and validated on AMD MI250X (ROCm 6.2, PyTorch 2.7.1, Triton 3.3) on the LUMI supercomputer; `slurm/`
has example jobs. The kernel is plain Triton and should run on NVIDIA Ampere or newer, where it autotunes its
launch configuration, but that path has not been tested: run `scripts/test_kernel.py` and the comparison in
[docs/ENGINE.md](docs/ENGINE.md) before trusting it. Supported: residual-stream (`resid_post`) JumpReLU SAEs,
Gemma Scope 2 widths 16k and 262k, Gemma 3 270M to 27B.

## Gallery of topic features

`gallery/site/` is a static page of 36 topic features of Gemma 3 27B, read by hand. A topic feature belongs to one
content domain and fires over spans of its texts, and it also fires in at least five other domains. The page shows,
for each feature, where it fires per domain and the passages it fires on, with every fired token highlighted.

It is published by the workflow in `.github/workflows/pages.yml` (repository settings, Pages, source "GitHub
Actions"). `gallery/build_gallery.py` rebuilds it; see step 4 of [thesis/README.md](thesis/README.md).

## Inspecting a capture

[notebooks/inspect_capture.ipynb](notebooks/inspect_capture.ipynb) reads any capture with
`saepipe.reader.Capture` and needs no GPU. It walks through:

- one prompt, token by token;
- the features a prompt is made of;
- where a feature fires in a text;
- the features specific to a domain;
- how much two domains share, layer by layer;
- where one feature fires across the capture.

## The thesis

The engine was built for the master's thesis of Joonatan M. Laato (2026) on how much of a model's SAE features are
shared between domains that differ in language, in content or in both. The findings are reported separately and
will be linked here. [thesis/README.md](thesis/README.md) describes the scripts:

- `thesis/data/` builds the 28-domain prompt set: five natural languages, four programming languages, SQL, assembly
  and eight English content domains ([thesis/data/README.md](thesis/data/README.md)).
- `thesis/analysis/` holds the analyses in run order, grouped into always-on features, feature classes, sharing
  between domains, and topic features.
- `thesis/analysis/classes/classifier/` holds the trained classifier that gives every feature one of five classes:
  position, token, word, concept or context.

## Licence

MIT (`LICENSE`). Gemma 3 and Gemma Scope 2 are subject to the Gemma terms of use. The prompt texts of the thesis
dataset, and the passages quoted in the gallery, come from third-party datasets under their own licences, listed
in `thesis/data/README.md`.
