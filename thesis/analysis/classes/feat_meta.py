#!/usr/bin/env python3
"""feat_meta.py (CPU) — per layer: the 28-domain corpus rate of every feature (agg_src.load) and its class (tf_class_v6, 255 = never fires).
Output work/feat_meta_27b.npz: rate float16 [62, 262144], cls uint8 [62, 262144]."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../lib"))  # thesis/analysis/lib
import numpy as np, sys; from agg_src import load, S
R = np.zeros((62, 262144), np.float16); C = np.full((62, 262144), 255, np.uint8)
for L in range(62):
    d = load("27b", L); R[L] = d["rate"]; C[L] = np.load(f"{S}/tf_class_v6/L{L}.npz")["cls"]; print(L, flush=True)
np.savez_compressed(f"{S}/feat_meta_27b.npz", rate=R, cls=C); print("FEAT_META_DONE")
