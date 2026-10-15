#!/usr/bin/env python3
"""ent_ctl2.py <size> — two controls of the pair sharing, from stats (28 domains, halves kept).
(a) same-source baseline: for the parallel-language pairs (A, B), Jaccard of A's odd-half set and B's even-half set
    (same source FLORES, other sentences, other language) beside A odd vs B odd (the same sentences) and A odd vs A even
    (same language, other sentences, same source); top-5,000 by half-mass and all fired; machinery (corpus rate >= .6) removed.
    The same for the parallel-code pairs.
(b) weighted Jaccard (Ruzicka): sum_f min(MASS_A, MASS_B) / sum_f max(MASS_A, MASS_B) over all non-machinery features, for
    every pair of the 28 domains, per layer.
Output work/ent_ctl2_{size}.json."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../lib"))  # thesis/analysis/lib
from paths import STATS
import sys, json, glob, re, numpy as np
from agg_src import load, S, LANGS, CODES
size = sys.argv[1]; LAYERS = sorted(int(re.search(r"L(\d+)\.npz$", p).group(1)) for p in glob.glob(f"{STATS}/262k/{size}/L*.npz"))
def topk(fired, MASS, K):
    f = np.where(fired)[0]; B = np.zeros_like(fired); B[f[np.argsort(-MASS[f], kind="stable")[:K]]] = True; return B
def J(a, b): return float((a & b).sum() / max((a | b).sum(), 1))
out = {"size": size, "layers": {}}
for L in LAYERS:
    d = load(size, L); names = d["names"]; rate = np.nan_to_num(d["rate"]); keep = rate < 0.6
    NFDh, MASSh, MASS = d["NFD_h"], d["MASS_h"], d["MASS"]
    rec = {"same_source": {}, "ruzicka": None, "ruzicka_par": None}
    for grp, pre, members in (("sentences", "par_lang_", LANGS), ("programs", "par_code_", CODES)):
        idx = [names.index(pre + m) for m in members]
        res = {"same content": {"top5000": [], "all": []}, "same source other content": {"top5000": [], "all": []}, "same language": {"top5000": [], "all": []}}
        sets = {}
        for i in idx:
            for h in (0, 1):
                fired = (NFDh[h, i] > 0) & keep; sets[(i, h, "all")] = fired; sets[(i, h, "top5000")] = topk(fired, MASSh[h, i], 5000)
        for a in idx:
            for b in idx:
                if a == b: continue
                for key in ("top5000", "all"):
                    res["same content"][key].append(J(sets[(a, 0, key)], sets[(b, 0, key)]))
                    res["same source other content"][key].append(J(sets[(a, 0, key)], sets[(b, 1, key)]))
            for key in ("top5000", "all"): res["same language"][key].append(J(sets[(a, 0, key)], sets[(a, 1, key)]))
        rec["same_source"][grp] = {k: {kk: float(np.mean(v)) for kk, v in vv.items()} for k, vv in res.items()}
    M = MASS[:, keep].astype(np.float64); n = M.shape[0]; R = np.zeros((n, n))
    for a in range(n):
        for b in range(a + 1, n):
            R[a, b] = R[b, a] = np.minimum(M[a], M[b]).sum() / max(np.maximum(M[a], M[b]).sum(), 1e-9)
    iu = np.triu_indices(n, 1); rec["ruzicka"] = float(R[iu].mean())
    par = [(names.index(f"par_lang_{a}"), names.index(f"par_lang_{b}")) for i, a in enumerate(LANGS) for b in LANGS[i + 1:]] + \
          [(names.index(f"par_code_{a}"), names.index(f"par_code_{b}")) for i, a in enumerate(CODES) for b in CODES[i + 1:]]
    rec["ruzicka_par"] = float(np.mean([R[a, b] for a, b in par])); rec["ruzicka_mat"] = R.tolist()
    out["layers"][str(L)] = rec; print(L, {k: round(v["top5000"], 3) for k, v in rec["same_source"]["sentences"].items()}, "ruz", round(rec["ruzicka"], 3), round(rec["ruzicka_par"], 3), flush=True)
out["names"] = names; json.dump(out, open(f"{S}/ent_ctl2_{size}.json", "w")); print("ENT_CTL2_DONE")
