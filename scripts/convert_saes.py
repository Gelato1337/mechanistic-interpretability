#!/usr/bin/env python3
"""
convert_saes.py — write encoder-only bf16 copies of Gemma Scope 2 SAEs (w_enc, b_enc, threshold) for fast loading.
For the 27B 262k SAEs this is 2.8 GB per layer instead of 11.3 GB (fp32 encoder + decoder); capture.py reads
them with --sae-bf16-dir. Optional: capture.py also loads the original files directly.

    python scripts/convert_saes.py --model 27b --layers all --out sae_bf16
"""
import argparse
from saepipe.sae import GemmaScope2

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--layers", default="all", help="'all' or comma-separated")
ap.add_argument("--n-layers", type=int, default=None, help="needed with --layers all (18/26/34/48/62 for 270m/1b/4b/12b/27b)")
ap.add_argument("--width", default="262k")
ap.add_argument("--out", required=True)
a = ap.parse_args()
N = {"270m": 18, "1b": 26, "4b": 34, "12b": 48, "27b": 62}
layers = range(a.n_layers or N[a.model]) if a.layers == "all" else [int(x) for x in a.layers.split(",")]
src = GemmaScope2(a.model, width=a.width, bf16_dir=a.out)
for L in layers:
    print(L, src.convert_bf16(L), flush=True)
