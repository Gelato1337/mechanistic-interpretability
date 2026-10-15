#!/usr/bin/env python3
"""supp_machinery.py (CPU) — the always-on features across all five model sizes and the 28 domains.
Inputs  the statistics of scripts/aggregate.py for every size (agg_src.load), per domain d and feature f: MASS[d,f] = mean over the
        domain's prompts of (sum of activations over tokens 2..n / tokens), NFD[d,f] = prompts fired, and the corpus rate
        r(f) = prompts fired / 13,986.
Machinery = r(f) >= .6 (the bridge_v4 rule). Per size, layer and domain:
        mach  = share of the domain's total activation mass carried by machinery features
        top1  = share carried by the single loudest feature,  top1_id = that feature
        nmach = machinery features at that layer (same for every domain), nfired = features firing in the domain
Output work/supp_machinery.json — small, plotted locally by v4_supp.py."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../lib"))  # thesis/analysis/lib
from paths import WORK, STATS
import os, sys, glob, re, json, numpy as np
from agg_src import load
S = WORK; MACH = 0.6
out = {}
for size in ["270m", "1b", "4b", "12b", "27b"]:
    files = {int(re.search(r"L(\d+)\.npz$", p).group(1)): p for p in glob.glob(f"{STATS}/262k/{size}/L*.npz")}
    if not files: print("skip", size, flush=True); continue
    rec = {"n_layers": None, "names": None, "layers": {}}
    for L in sorted(files):
        z = load(size, L, "262k"); names = z["names"]; MASS = np.asarray(z["MASS"], np.float64); NFD = np.asarray(z["NFD"]); rate = np.asarray(z["rate"], np.float64)
        rec["names"] = names; rec["n_layers"] = int(z["n_layers"]) if "n_layers" in z else 62
        mach = rate >= MACH; tot = np.maximum(MASS.sum(1), 1e-12); top = MASS.argmax(1)
        rec["layers"][L] = {"mach": (MASS[:, mach].sum(1) / tot).round(4).tolist(),
                            "top1": (MASS[np.arange(len(names)), top] / tot).round(4).tolist(),
                            "top1_id": top.tolist(), "top1_is_mach": mach[top].tolist(),
                            "nmach": int(mach.sum()), "nfired": (NFD > 0).sum(1).tolist()}
        print(size, L, "mach mass share median", round(float(np.median(rec["layers"][L]["mach"])), 3), "nmach", int(mach.sum()), flush=True)
    out[size] = rec
json.dump(out, open(f"{S}/supp_machinery_v2.json", "w")); print("SUPP_MACHINERY_DONE")
