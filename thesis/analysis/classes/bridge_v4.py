#!/usr/bin/env python3
"""bridge_v4.py <L0> <L1> (CPU; layers L0..L1-1, e.g. 8 tasks of 8 layers) — the bridge with the judge-calibrated TF classifier instead of the hand rule.
Step 0: the classifier of feature_classifier.py (the shipped model, or with RETRAIN=1 a fit on work/tf_train.json
{id: {"y": class, "x": {stat: value}}} written locally by tf_train_export.py from the Sonnet labels (v3_labels.json) and, when
present, the Qwen labels of the domain-wide sample — classes position/token/word/concept/context.
Per layer: X for every fired feature from work/tf_feats/L{L}.npz (all stats + layer), predicted class and its
probability saved to work/tf_class{TAG}/L{L}.npz (cls uint8 in CLS order, 255 = never fires; prob float16).
Bands kept outside the classifier, first match wins: machinery rate >= .6, rare nf < 30 (too few fires for the statistics).
Sets: unbounded and top-K by MASS[d] for K in KS. Per pair (a, b): union, shared, mach_union, machinery, rare, then the five
classes of the shared features excluding machinery and rare, and rare_{class} = the five classes of the rare shared features
(so class + rare_class = every non-machinery shared feature of that class). Output work/bridge_v4{TAG}/L{L}.json in the
bridge_v3 layout (env TAG; the classifier probabilities are prevalence-matched, tf_balance.py recipe C).
A layer whose tf_class{TAG}/L{L}.npz exists reuses those labels
(no retraining); env FORCE=1 rewrites existing L{L}.json."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../lib"))  # thesis/analysis/lib
from paths import WORK
import sys, os, json, time, numpy as np
SRC = os.environ.get("SRC", "v2")   # "v2" (the thesis): MASS / NFD / nf / rate from stats (scripts/aggregate.py: JumpReLU fixed, duplicates dropped); classes unchanged
if SRC: from agg_src import load as agg_load
S = WORK; D = 262144
CLS = ["position", "token", "word", "concept", "context"]; KS = [100, 1000, 2000, 5000, 10000, 20000]; MACH = 0.6; RARE = 30
TAG = os.environ.get("TAG", "_v6")   # "_v6" (the thesis) = Sonnet + Qwen-v6 domain-wide rows, prevalence-matched
OUT = f"{S}/bridge_v4{TAG}" + ("_" + SRC if SRC else ""); os.makedirs(OUT, exist_ok=True); os.makedirs(f"{S}/tf_class{TAG}", exist_ok=True)
L0, L1 = int(sys.argv[1]), int(sys.argv[2]); FORCE = os.environ.get("FORCE") == "1"
need_train = any(not os.path.exists(f"{S}/tf_class{TAG}/L{L}.npz") for L in range(L0, L1))
if need_train:   # the shipped model (feature_classifier.py); retrained from work/tf_train.json only with RETRAIN=1
  import feature_classifier as FC
  MODEL = FC.train(f"{S}/tf_train.json") if os.environ.get("RETRAIN") == "1" or not os.path.exists(FC.MODEL) else FC.load()
for L in range(L0, L1):
    if os.path.exists(f"{OUT}/L{L}.json") and not FORCE: print("skip", L, flush=True); continue
    t0 = time.time(); z = np.load(f"{S}/tf_feats/L{L}.npz", allow_pickle=True); names = list(z["names"]); n = len(names); MASS = z["MASS"]; NFD = z["NFD"]
    nf = z["nf"]; fired = ~np.isnan(nf); idx = np.where(fired)[0]
    if SRC == "v2":
        a = agg_load("27b", L, "262k"); names = a["names"]; n = len(names); MASS = a["MASS"]; NFD = a["NFD"]   # the 28 domains; corpus = their 13,986 prompts
        fired = a["nf"] > 0; idx = np.where(fired)[0]; nf = np.where(fired, a["nf"], np.nan).astype(np.float32); z = dict(z); z["rate"] = a["rate"]
    if os.path.exists(f"{S}/tf_class{TAG}/L{L}.npz"):
        cls = np.load(f"{S}/tf_class{TAG}/L{L}.npz")["cls"]
        if SRC:   # a few features fire in the v2 capture that never fired in the capture the labels were made on (bf16 numerics at the threshold): no label -> left out, counted
            nolab = fired & (cls == 255); print(f"L{L}: {int(nolab.sum())} of {int(fired.sum())} fired features have no class label and are left out", flush=True)
            fired = fired & ~nolab; idx = np.where(fired)[0]; NFD = NFD * fired[None, :]
        else: assert ((cls != 255) == fired).all()
        yhat = cls[idx].astype(int)
    else:
        cls, prob = FC.predict(MODEL, np.load(f"{S}/tf_feats/L{L}.npz", allow_pickle=True), L); yhat = cls[idx].astype(int)
        np.savez_compressed(f"{S}/tf_class{TAG}/L{L}.npz", cls=cls, prob=prob)
    rate = np.nan_to_num(z["rate"]); mach = rate >= MACH; rare = ~mach & (np.nan_to_num(nf) < RARE) & fired
    rec = {"names": names, "ks": KS, "n_all": int(fired.sum()), "classes": CLS, "n_class": np.bincount(yhat, minlength=5).tolist(),
           "n_class_core": np.bincount(yhat[~mach[idx] & ~rare[idx]], minlength=5).tolist(), "n_class_rare": np.bincount(yhat[rare[idx]], minlength=5).tolist()}
    sets = {"unbounded": [np.where(NFD[i] > 0)[0] for i in range(n)]}
    for K in KS: sets[f"top{K}"] = [(lambda f: f[np.argsort(-MASS[i, f], kind="stable")[:K]])(np.where(NFD[i] > 0)[0]) for i in range(n)]
    for key, SS in sets.items():
        keys = ["union", "shared", "mach_union", "machinery", "rare"] + CLS + [f"rare_{k}" for k in CLS]; m = {k: np.zeros((n, n), int) for k in keys}
        B = np.zeros((n, D), bool)
        for i in range(n): B[i, SS[i]] = True
        for a in range(n):
            for b in range(a, n):
                sh = B[a] & B[b]; un = B[a] | B[b]; core = sh & ~mach & ~rare
                c = {"union": un.sum(), "shared": sh.sum(), "mach_union": (un & mach).sum(), "machinery": (sh & mach).sum(), "rare": (sh & rare).sum()}
                cc = np.bincount(cls[core], minlength=256)[:5]; cr = np.bincount(cls[sh & rare], minlength=256)[:5]
                for j, k in enumerate(CLS): c[k] = cc[j]; c[f"rare_{k}"] = cr[j]
                for k, v in c.items(): m[k][a, b] = m[k][b, a] = int(v)
        rec[key] = {k: v.tolist() for k, v in m.items()}
    json.dump(rec, open(f"{OUT}/L{L}.json", "w")); print(f"L{L}: {time.time()-t0:.0f}s classes {rec['n_class']}", flush=True)
print("BRIDGE_V4_DONE")
