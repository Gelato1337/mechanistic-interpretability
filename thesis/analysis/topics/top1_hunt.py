#!/usr/bin/env python3
"""top1_hunt.py — the top-1 string share of every feature (share of its firings on its most frequent token string; tf_feats.py)
at the eight 27B layers of the topic-feature analysis, collected for fig_ladder.py. Output work/top1_hunt_27b.npz {top1_<L>}."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../lib"))  # thesis/analysis/lib
import numpy as np
from paths import WORK
LAYERS = [31, 35, 39, 43, 47, 51, 55, 59]
np.savez_compressed(f"{WORK}/top1_hunt_27b.npz", **{f"top1_{L}": np.load(f"{WORK}/tf_feats/L{L}.npz")["top1"].astype(np.float32) for L in LAYERS})
print("TOP1_DONE")
