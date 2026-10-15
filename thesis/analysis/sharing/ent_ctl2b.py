#!/usr/bin/env python3
"""ent_ctl2b.py <size> — the Wikipedia baseline on the same half sets as ent_ctl2.py, so the two baselines are comparable:
J(par_lang_A odd, div_lang_B odd) for A != B (other language, other content, other source), top-5,000 by half-mass and all fired,
machinery removed. Output work/ent_ctl2b_{size}.json."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../lib"))  # thesis/analysis/lib
from paths import STATS, DIVERSE_LANG
import sys, json, glob, re, numpy as np
from agg_src import load, S, LANGS
size = sys.argv[1]; LAYERS = sorted(int(re.search(r"L(\d+)\.npz$", p).group(1)) for p in glob.glob(f"{STATS}/262k/{size}/L*.npz"))
DIV = dict(zip(LANGS, DIVERSE_LANG))
def topk(fired, MASS, K):
    f = np.where(fired)[0]; B = np.zeros_like(fired); B[f[np.argsort(-MASS[f], kind="stable")[:K]]] = True; return B
def J(a, b): return float((a & b).sum() / max((a | b).sum(), 1))
out = {"size": size, "layers": {}}
for L in LAYERS:
    d = load(size, L); names = d["names"]; rate = np.nan_to_num(d["rate"]); keep = rate < 0.6; NFDh, MASSh = d["NFD_h"], d["MASS_h"]
    sets = {}
    for n in [f"par_lang_{l}" for l in LANGS] + list(DIV.values()):
        i = names.index(n); fired = (NFDh[0, i] > 0) & keep; sets[(n, "all")] = fired; sets[(n, "top5000")] = topk(fired, MASSh[0, i], 5000)
    res = {"top5000": [], "all": []}
    for a in LANGS:
        for b in LANGS:
            if a == b: continue
            for key in res: res[key].append(J(sets[(f"par_lang_{a}", key)], sets[(DIV[b], key)]))
    out["layers"][str(L)] = {k: float(np.mean(v)) for k, v in res.items()}; print(L, out["layers"][str(L)], flush=True)
json.dump(out, open(f"{S}/ent_ctl2b_{size}.json", "w")); print("ENT_CTL2B_DONE")
