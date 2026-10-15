#!/usr/bin/env python3
"""dom_sel.py (CPU) — the domain-wide judge sample: features drawn from EVERY domain, not only the parallel sets,
so the heuristic classifier is tested where the bridge is measured.
Per layer in SL and per domain d (33): features that fire in d ranked by MASS[d] (tf_feats.py, the bridge_v3 top-K rank);
strata by rank: top100 (< 100), top1000 (100..999), top5000 (1000..4999), tail (>= 5000); NPER features per stratum, drawn at
random (seed 7), features already judged (work/v3_labels.json) excluded. Env LAYERS, NPER, OUT; the 27 display
domains only (div_lang_* and disjoint_lang_swedish dropped).
Output work/dom_sel.json {L: {d: [{f, rank, stratum, mass, nfd}]}}."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../lib"))  # thesis/analysis/lib
from paths import WORK, UNUSED_DIVERSE, SWEDISH_DIVERSE
import json, re, os, numpy as np
S = WORK
SL = [int(x) for x in os.environ.get("LAYERS", "0,6,12,18,24,32,43,52,61").split(",")]; NPER = int(os.environ.get("NPER", "2")); OUTF = os.environ.get("OUT", f"{S}/dom_sel.json")
DROP = UNUSED_DIVERSE | {SWEDISH_DIVERSE}   # the 27 display domains (no Swedish diverse-language domain)
STRATA = [("top100", 0, 100), ("top1000", 100, 1000), ("top5000", 1000, 5000), ("tail", 5000, 10**9)]
rng = np.random.default_rng(7); done = {}
for k in json.load(open(f"{S}/v3_labels.json")):
    m = re.match(r".*_L(\d+)_f(\d+)$", k); done.setdefault(int(m.group(1)), set()).add(int(m.group(2)))
out = {}; ntot = 0
for L in SL:
    z = np.load(f"{S}/tf_feats/L{L}.npz", allow_pickle=True); MASS = z["MASS"]; NFD = z["NFD"]; names = list(z["names"]); out[L] = {}
    for i, d in enumerate(names):
        if d in DROP: continue
        fs = np.where(NFD[i] > 0)[0]; order = fs[np.argsort(-MASS[i, fs], kind="stable")]; rows = []
        for name, lo, hi in STRATA:
            pool = [(r, int(f)) for r, f in enumerate(order[lo:hi], lo) if int(f) not in done.get(L, ())]
            for j in rng.choice(len(pool), min(NPER, len(pool)), replace=False):
                r, f = pool[j]; rows.append({"f": f, "rank": r, "n": int(len(order)), "stratum": name, "mass": float(MASS[i, f]), "nfd": int(NFD[i, f])})
        out[L][d] = rows; ntot += len(rows)
    print(L, ntot, flush=True)
json.dump(out, open(OUTF, "w"), indent=0); print("DOM_SEL_DONE", ntot)
