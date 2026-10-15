#!/usr/bin/env python3
"""
compare_captures.py — compare two capture directories layer by layer (same prompts, same layers): firing sets
(Jaccard, firings only in A / only in B) and activation values on the shared firings.

    python scripts/capture.py --model 1b --prompts prompts.csv --n-per-domain 4 --layers 0,12,25 --out runs/a
    python scripts/capture.py --model 1b --prompts prompts.csv --n-per-domain 4 --layers 0,12,25 --encoder torch --out runs/b
    python scripts/compare_captures.py runs/a runs/b

The Triton encode with the fp32 epilogue and the torch reference agree up to firings whose pre-activation lies
within bf16 rounding of the threshold; anything else is a bug.
"""
import sys, json
import numpy as np
from pathlib import Path
from saepipe.io import load_layer

a, b = Path(sys.argv[1]), Path(sys.argv[2])
ta, tb = np.load(a / "tokens.npz"), np.load(b / "tokens.npz")
assert (ta["n_tokens"] == tb["n_tokens"]).all() and (ta["prompt_ids"] == tb["prompt_ids"]).all(), "different prompt sets"
layers = sorted(int(p.name.split("_")[1].split(".")[0]) for p in a.glob("layer_*"))
N = 1 << 20
for L in layers:
    ra, fa, va = load_layer(a, L); rb, fb, vb = load_layer(b, L)
    ka, kb = ra.astype(np.int64) * N + fa, rb.astype(np.int64) * N + fb
    c, ia, ib = np.intersect1d(ka, kb, return_indices=True)
    d = np.abs(va[ia] - vb[ib]) / np.maximum(np.abs(va[ia]), 1e-6) if len(c) else np.zeros(1)
    print(json.dumps(dict(layer=L, firings_a=len(ka), firings_b=len(kb), only_a=len(ka) - len(c), only_b=len(kb) - len(c),
                          jaccard=round(len(c) / max(1, len(ka) + len(kb) - len(c)), 6), max_rel_value_diff=float(d.max()))))
