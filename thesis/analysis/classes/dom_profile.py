#!/usr/bin/env python3
"""dom_profile.py (CPU, one task for all 62 layers) — what a single domain looks like on its own, per layer:
the class make-up of the features that fire in the domain's prompts, in three weightings.
  count  every feature that fires in the domain, one vote each
  prompts  weighted by NFD[d, f] = number of the domain's 500 prompts the feature fires in
  mass   weighted by MASS[d, f] = mean over the domain's prompts of the feature's activation mass per token (tokens 2..n) —
         this is the "average prompt" view: the share of an average prompt's total feature activation that goes to each class
Classes from work/tf_class{TAG}/L{L}.npz (bridge_v4.py; cls 255 = never fires); machinery (corpus rate >= .6) and rare
(< 30 corpus fires) are kept as their own bands, before the five classes, so the shares sum to 1 over 7 bands.
Output work/dom_profile{TAG}.json {L: {domain: {"count"|"prompts"|"mass": [7 shares], "n": fired features, "nfd_sum", "mass_sum"}}}
and bands = ["machinery", "rare", position, token, word, concept, context].
2026-09-07: "rare_count" / "rare_prompts" / "rare_mass" = the rare band split by the classifier's five classes (shares of the
same total, so they sum to the rare band); folding them into the classes gives the six-band view machinery + five classes."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../lib"))  # thesis/analysis/lib
from paths import WORK
import os, sys, json, time, numpy as np
SRC = os.environ.get("SRC", "v2")   # "v2" (the thesis): MASS / NFD / nf / rate from stats (scripts/aggregate.py: JumpReLU fixed, duplicates dropped); classes unchanged
if SRC: from agg_src import load as agg_load
S = WORK; TAG = os.environ.get("TAG", "_v6")
CLS = ["position", "token", "word", "concept", "context"]; BANDS = ["machinery", "rare"] + CLS; MACH = 0.6; RARE = 30
out = {}
for L in range(62):
    t0 = time.time(); z = np.load(f"{S}/tf_feats/L{L}.npz", allow_pickle=True); names = list(z["names"]); MASS = z["MASS"]; NFD = z["NFD"]; nf = z["nf"]; rate = np.nan_to_num(z["rate"])
    if SRC == "v2":
        a = agg_load("27b", L, "262k"); names = a["names"]; MASS = a["MASS"]; NFD = a["NFD"]; nf = np.where(a["nf"] > 0, a["nf"], np.nan); rate = a["rate"]   # the 28 domains
    cls = np.load(f"{S}/tf_class{TAG}/L{L}.npz")["cls"]; fired = (cls != 255) & (np.nan_to_num(nf) > 0) if SRC else cls != 255
    band = np.full(len(cls), -1, int); band[fired] = cls[fired] + 2
    mach = fired & (rate >= MACH); rare = fired & ~mach & (np.nan_to_num(nf) < RARE); band[mach] = 0; band[rare] = 1
    out[L] = {}
    for i, d in enumerate(names):
        m = (NFD[i] > 0) & fired; b = band[m]; rec = {"n": int(m.sum()), "nfd_sum": float(NFD[i][m].sum()), "mass_sum": float(MASS[i][m].sum())}
        for key, w in (("count", np.ones(m.sum())), ("prompts", NFD[i][m].astype(float)), ("mass", MASS[i][m].astype(float))):
            s = np.bincount(b, weights=w, minlength=7); tot = max(s.sum(), 1e-9); rec[key] = (s / tot).round(5).tolist()
            r = rare[m]; rec[f"rare_{key}"] = (np.bincount(cls[m][r], weights=w[r], minlength=5)[:5] / tot).round(5).tolist()
        out[L][d] = rec
    print(L, f"{time.time() - t0:.0f}s", flush=True)
json.dump({"bands": BANDS, "layers": out}, open(f"{S}/dom_profile{TAG}" + ("_" + SRC if SRC else "") + ".json", "w")); print("DOM_PROFILE_DONE")
