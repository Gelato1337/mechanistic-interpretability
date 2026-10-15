"""
saepipe — fast sparse-autoencoder feature extraction for large decoder LMs.

Building blocks (see docs/ENGINE.md for what is generic and what is specific):
  backend.py        GPU vendor detection (ROCm / CUDA) + capability checks          [generic]
  kernels/          fused JumpReLU encode in Triton + per-vendor launch configs     [generic code, vendor-tuned configs]
  sae.py            SAE weight sources: JumpReLUEncoder + GemmaScope2 adapter       [interface generic, adapter Gemma-Scope-2]
  model.py          HF decoder loading, layer->device map, pipeline stages          [generic for HF decoders with .layers]
  encode.py         encoders with one interface: TritonJumpReLU (async), TorchJumpReLU (sync)  [generic]
  pipeline.py       forward-only GPipe scheduler over GPUs, calls the encoder per layer        [generic; resid_post only]
  io.py             streaming shard writer, shard reader                            [generic]
  reader.py         Capture: a capture directory as prompts, tokens and per-layer codes  [generic]
"""
