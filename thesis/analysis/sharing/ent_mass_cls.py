#!/usr/bin/env python3
"""ent_mass_cls.py [size] — the weight of the shared features, by class. For every layer, set rule (top-1,000 / top-5,000 by mass / all fired;
machinery at corpus rate >= .6 removed) and ordered pair (A, B): the share of A's mass (non-machinery total) that lies in features of A's set
that are also in B's set, split by the classifier class of the feature (tf_class_v6/L{L}.npz, position/token/word/concept/context; rare
features carry their predicted class as everywhere in the thesis). Same set rule and denominator as ent_mass.py, so the five classes sum
to ent_mass w_shared (before its symmetrisation). Output work/ent_mass_cls_{size}.json: layers -> K -> class -> n x n list."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../lib"))  # thesis/analysis/lib
from paths import STATS
import sys, os, json, glob, re, numpy as np
from agg_src import load, S
size = sys.argv[1] if len(sys.argv) > 1 else "27b"; CLS = ["position", "token", "word", "concept", "context"]
LAYERS = sorted(int(re.search(r"L(\d+)\.npz$", p).group(1)) for p in glob.glob(f"{STATS}/262k/{size}/L*.npz"))
out = {"size": size, "classes": CLS, "ks": ["1000", "5000", "all"], "layers": {}}
for L in LAYERS:
    d = load(size, L); names = d["names"]; MASS = d["MASS"].astype(np.float64); fired = d["NFD"] > 0; keep = np.nan_to_num(d["rate"]) < 0.6
    cls = np.load(f"{S}/tf_class_v6/L{L}.npz")["cls"]; n = len(names); tot = (MASS * keep).sum(1); rec = {}
    for K in [1000, 5000, "all"]:
        B = np.zeros_like(fired)
        for i in range(n):
            f = np.where(fired[i] & keep)[0]
            if K == "all": B[i, f] = True
            else: B[i, f[np.argsort(-MASS[i, f], kind="stable")[:K]]] = True
        Wm = MASS * B; Bf = B.astype(np.float64); rec[str(K)] = {}
        for j, c in enumerate(CLS):
            M = (Wm * (cls == j)) @ Bf.T / tot[:, None]; np.fill_diagonal(M, np.nan); rec[str(K)][c] = np.round(M, 5).tolist()
        other = (Wm * (cls > 4)) @ Bf.T / tot[:, None]; np.fill_diagonal(other, np.nan); rec[str(K)]["unclassed"] = np.round(other, 5).tolist()
    out["layers"][str(L)] = rec; out["names"] = names
    iu = np.triu_indices(n, 1); print(L, {K: round(float(sum(np.array(rec[K][c])[iu].mean() for c in CLS)), 3) for K in rec}, flush=True)
json.dump(out, open(f"{S}/ent_mass_cls_{size}.json", "w")); print("ENT_MASS_CLS_DONE")
