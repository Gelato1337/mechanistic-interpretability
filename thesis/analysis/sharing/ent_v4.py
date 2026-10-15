#!/usr/bin/env python3
"""ent_v4.py <size> — the domain-pair Jaccard of every layer for one model size, from stats (agg_src.py; JumpReLU fixed,
philosophy duplicates dropped). env WIDTH=262k|16k, EXTRA=lenm (append the length-matched domains when their file exists).
Sets per domain: fired = NFD > 0; top-K = the K fired features with the largest MASS (K in KS) and top10pct = the largest 10 % of the
domain's fired features; machinery = corpus rate >= thr for thr in MACHS (rate over the 33 base domains).
Per layer: J[thr][key] = |A&B - machinery| / |A|B - machinery|; J_all[key] plain; J_half[key] = [J on odd prompts, J on even prompts]
(machinery .6). Output work/ent_v4_{width}_{size}.json {"size","width","ks","names","n_layers","layers":{L:{...}}}"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../lib"))  # thesis/analysis/lib
from paths import STATS, stats_file
import sys, os, json, time, glob, re, numpy as np
from agg_src import load, S
size = sys.argv[1]; WIDTH = os.environ.get("WIDTH", "262k"); EXTRA = os.environ.get("EXTRA", "lenm")
KS = [100, 1000, 2000, 5000, 10000, 20000] if WIDTH == "262k" else [6, 62, 125, 312, 625, 1250]; MACHS = [0.5, 0.6, 0.8]
LAYERS = sorted(int(re.search(r"L(\d+)\.npz$", p).group(1)) for p in glob.glob(f"{STATS}/{WIDTH}/{size}/L*.npz"))
extra_ok = lambda L: EXTRA and os.path.exists(stats_file(size, L, WIDTH, EXTRA))
def jac(B):
    a = B.sum(1).astype(np.int64); Bf = B.astype(np.float32); sh = (Bf @ Bf.T).astype(np.int64); un = a[:, None] + a[None, :] - sh
    return (sh / np.maximum(un, 1)).tolist()
def topk(fired, MASS, K):
    B = np.zeros_like(fired)
    for i in range(fired.shape[0]):
        f = np.where(fired[i])[0]; k = K if isinstance(K, int) else max(1, int(round(K * len(f)))); B[i, f[np.argsort(-MASS[i, f], kind="stable")[:k]]] = True
    return B
out = {"size": size, "width": WIDTH, "ks": KS, "machs": MACHS, "names": None, "n_layers": None, "layers": {}}
for L in LAYERS:
    t0 = time.time(); d = load(size, L, WIDTH, EXTRA if extra_ok(L) else None); names = d["names"]; MASS = d["MASS"]; NFD = d["NFD"]; rate = np.nan_to_num(d["rate"])
    out["names"] = names; out["n_layers"] = int(d["n_layers"]); fired = NFD > 0
    sets = {"unbounded": fired, "top10pct": topk(fired, MASS, 0.10)}
    for K in KS: sets[f"top{K}"] = topk(fired, MASS, K)
    rec = {"n_mach": {str(t): int((rate >= t).sum()) for t in MACHS}, "n_fired": fired.sum(1).tolist(), "J": {}, "J_all": {}, "J_half": {}}
    for t in MACHS:
        keep = rate < t; rec["J"][str(t)] = {key: jac(B & keep) for key, B in sets.items()}
    rec["J_all"] = {key: jac(B) for key, B in sets.items()}
    keep = rate < 0.6
    for h in (0, 1):
        fh = d["NFD_h"][h] > 0; sh = {"unbounded": fh, "top10pct": topk(fh, d["MASS_h"][h], 0.10)}
        for K in KS: sh[f"top{K}"] = topk(fh, d["MASS_h"][h], K)
        for key, B in sh.items(): rec["J_half"].setdefault(key, []).append(jac(B & keep))
    out["layers"][L] = rec; print(f"L{L}: {time.time() - t0:.0f}s domains {len(names)} fired {fired.sum(1).min()}-{fired.sum(1).max()} machinery {rec['n_mach']}", flush=True)
json.dump(out, open(f"{S}/ent_v4_{WIDTH}_{size}.json", "w")); print("ENT_V4_DONE")
