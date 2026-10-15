#!/usr/bin/env python3
"""ent_mass.py <size> [WIDTH] — how much the shared features weigh. For every pair (A, B), layer and set rule (top-100 / top-1,000 /
top-5,000 by mass / all fired; machinery at corpus rate >= .6 removed), the share of A's mass that lies in A's set and in the
intersection with B's set: w_set[A] = sum_{f in S_A} m_A[f] / sum_f m_A[f]; w_shared[A,B] = sum_{f in S_A & S_B} m_A[f] / sum_f m_A[f].
Pair values are the mean of the two directions. Output work/ent_mass_{width}_{size}.json."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../lib"))  # thesis/analysis/lib
from paths import STATS
import sys, os, json, glob, re, numpy as np
from agg_src import load, S
size = sys.argv[1]; WIDTH = sys.argv[2] if len(sys.argv) > 2 else "262k"
KS = [100, 1000, 5000] if WIDTH == "262k" else [100, 1000]
LAYERS = sorted(int(re.search(r"L(\d+)\.npz$", p).group(1)) for p in glob.glob(f"{STATS}/{WIDTH}/{size}/L*.npz"))
out = {"size": size, "width": WIDTH, "ks": KS + ["all"], "layers": {}}
for L in LAYERS:
    d = load(size, L, WIDTH); names = d["names"]; MASS = d["MASS"].astype(np.float64); fired = (d["NFD"] > 0); keep = np.nan_to_num(d["rate"]) < 0.6
    n = len(names); tot = (MASS * keep).sum(1); rec = {"w_set": {}, "w_shared": {}, "J": {}}
    for K in KS + ["all"]:
        B = np.zeros_like(fired)
        for i in range(n):
            f = np.where(fired[i] & keep)[0]
            if K == "all": B[i, f] = True
            else: B[i, f[np.argsort(-MASS[i, f], kind="stable")[:K]]] = True
        Wm = MASS * B                       # mass of A's set features
        rec["w_set"][str(K)] = (Wm.sum(1) / tot).tolist()
        M = np.zeros((n, n))
        for a in range(n):
            for b in range(n):
                if a != b: M[a, b] = (Wm[a] * B[b]).sum() / tot[a]
        rec["w_shared"][str(K)] = ((M + M.T) / 2).tolist()
        Bf = B.astype(np.float32); sh = Bf @ Bf.T; cnt = B.sum(1).astype(np.float64); un = cnt[:, None] + cnt[None, :] - sh
        rec["J"][str(K)] = (sh / np.maximum(un, 1)).tolist()
    out["layers"][str(L)] = rec; out["names"] = names
    iu = np.triu_indices(n, 1); print(L, {K: round(float(np.array(rec["w_shared"][str(K)])[iu].mean()), 3) for K in rec["w_shared"]}, flush=True)
json.dump(out, open(f"{S}/ent_mass_{WIDTH}_{size}.json", "w")); print("ENT_MASS_DONE")
