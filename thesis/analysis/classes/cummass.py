#!/usr/bin/env python3
"""cummass.py (CPU) — firing vs impact per domain: rank every feature that fires in a domain by its mass (mean activation
per token over tokens 2..n of an average prompt; the decoder rows are unit-norm, so mass = length written into the residual stream)
and record the cumulative share of the domain's total mass at fixed ranks, plus how many of the fired features are seen in < 30 of the
16,500 corpus prompts and what share of the mass they carry. Machinery (corpus rate >= .6) removed first.
Input work/tf_feats/L{L}.npz (27B). Output work/cummass.json {L: {domain: {n, ranks, cum, n_rare, mass_rare}}}."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../lib"))  # thesis/analysis/lib
from paths import WORK
import json, sys, numpy as np
from agg_src import load
S = WORK; RANKS = [1, 10, 100, 1000, 2000, 5000, 10000, 20000, 50000]
out = {}
for L in [2, 12, 24, 42, 61]:
    z = load("27b", L, "262k"); names = z["names"]; MASS = np.asarray(z["MASS"], np.float64); NFD = np.asarray(z["NFD"]); nf = np.asarray(z["nf"], float); rate = np.asarray(z["rate"], float)
    keep = rate < 0.6; out[L] = {}
    for i, d in enumerate(names):
        f = np.where((NFD[i] > 0) & keep)[0]; m = MASS[i, f]; o = np.argsort(-m); ms = m[o]; cum = np.cumsum(ms) / ms.sum(); rare = nf[f][o] < 30
        out[L][d] = {"n": int(len(f)), "ranks": [r for r in RANKS if r <= len(f)], "cum": [float(cum[r - 1]) for r in RANKS if r <= len(f)],
                     "n_rare": int(rare.sum()), "mass_rare": float(ms[rare].sum() / ms.sum()), "rare_beyond_5000": float(rare[5000:].mean()) if len(f) > 5000 else None}
    print(L, "done", flush=True)
json.dump(out, open(f"{S}/cummass_v2.json", "w")); print("CUMMASS_DONE")
