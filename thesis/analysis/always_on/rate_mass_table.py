#!/usr/bin/env python3
"""rate_mass_table.py — share of an average prompt's mass by corpus firing-rate band, every layer of Gemma 3 27B (stats 262k):
mass of a feature = mean over the 28 domains of its per-domain mass; bands by the share of the 13,986 prompts the feature fires in.
Output work/rate_mass_27b.json: per layer, number of features and share of total mass per band, and the loudest feature's share."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../lib"))  # thesis/analysis/lib
from paths import STATS
import sys, json, glob, re, numpy as np
from agg_src import load, S
BANDS = [(.8, 1.01), (.6, .8), (.5, .6), (.3, .5), (.1, .3), (.01, .1), (0, .01)]; out = {}
LAYERS = sorted(int(re.search(r"L(\d+)\.npz$", p).group(1)) for p in glob.glob(f"{STATS}/262k/27b/L*.npz"))
for L in LAYERS:
    d = load("27b", L); m = d["MASS"].astype(np.float64).mean(0); rate = np.nan_to_num(d["rate"]); tot = m.sum(); top = int(np.argmax(m))
    out[str(L)] = {"bands": [{"lo": lo, "hi": min(hi, 1.0), "n": int(((rate >= lo) & (rate < hi) & (m > 0)).sum()), "mass_share": float(m[(rate >= lo) & (rate < hi)].sum() / tot)} for lo, hi in BANDS],
                   "top_feature": top, "top_share": float(m[top] / tot), "top_rate": float(rate[top]), "n_fired": int((m > 0).sum())}
    print(L, [round(b["mass_share"], 3) for b in out[str(L)]["bands"]], "top", top, round(out[str(L)]["top_share"], 2), flush=True)
json.dump(out, open(f"{S}/rate_mass_27b.json", "w")); print("RATE_MASS_DONE")
