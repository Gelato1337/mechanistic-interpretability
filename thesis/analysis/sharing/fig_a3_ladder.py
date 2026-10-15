#!/usr/bin/env python3
"""fig_a3_ladder.py — the K ladder per size (most entangled pair, mean over all 378 pairs, least entangled pair), from work/
ent_v4_262k_{size}.json, in the thesis style: panel titles "kind · size", x label "layer", no in-image header. Output figures/act3/a3_ladder.png."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../lib"))  # thesis/analysis/lib
from paths import WORK, FIGS, UNUSED_DIVERSE
import notitle, json, os, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.ticker
S = WORK; SIZES = ["270m", "1b", "4b", "12b", "27b"]; SL = {"270m": "270M", "1b": "1B", "4b": "4B", "12b": "12B", "27b": "27B"}
LANGS = ["english", "finnish", "french", "german", "swedish"]; DROP = UNUSED_DIVERSE
KL3 = ["top100", "top1000", "top2000", "top5000", "top10000", "top20000", "unbounded"]; KLL = {"top100": "top-100", "top1000": "top-1,000", "top2000": "top-2,000", "top5000": "top-5,000", "top10000": "top-10,000", "top20000": "top-20,000", "unbounded": "all fired"}
KCOL = dict(zip(KL3, ["#9ccb3b", "#3a9d5d", "#1f8a80", "#1d5c8a", "#3b3f8f", "#5a2d7a", "#222222"]))
short = lambda d: d.replace("par_lang_", "FLORES ").replace("disjoint_lang_", "Wikipedia ").replace("div_lang_", "Wikipedia ").replace("par_code_", "Rosetta ").replace("div_code_", "Stack ").replace("code_", "").replace("_", " ")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(len(SIZES), 3, figsize=(14, 3.1 * len(SIZES)), squeeze=False)
for r, s_ in enumerate(SIZES):
    E = json.load(open(f"{S}/ent_v4_262k_{s_}.json")); nm = E["names"]; keep = [i for i, d in enumerate(nm) if d not in DROP and not d.startswith("lenm_")]; Ls = sorted(int(l) for l in E["layers"])
    def J(L, key):
        r = E["layers"][str(L)]; M = np.array(r["J"]["0.6"][key] if isinstance(r["J"].get("0.6"), dict) else r["J"][key], float)[np.ix_(keep, keep)]; np.fill_diagonal(M, np.nan); return M
    n3 = len(keep); iu = np.triu_indices(n3, 1); avg = np.nanmean([J(L, "unbounded") for L in Ls], 0)[iu]; nm3 = [nm[i] for i in keep]
    hi = int(np.argmax(avg)); lo = int(np.argmin(avg)); ph = (iu[0][hi], iu[1][hi]); pl = (iu[0][lo], iu[1][lo])
    fns = [lambda key: [J(L, key)[ph] for L in Ls], lambda key: [np.nanmean(J(L, key)[iu]) for L in Ls], lambda key: [J(L, key)[pl] for L in Ls]]
    titles = [f"most entangled pair · {SL[s_]}", f"mean of 378 pairs · {SL[s_]}", f"least entangled pair · {SL[s_]}"]
    for c in range(3):
        ax = axes[r, c]
        for key in KL3: ax.plot(Ls, fns[c](key), color=KCOL[key], ls="--" if key == "unbounded" else "-", lw=1.8 if key in ("top5000", "unbounded") else 1.1, label=KLL[key])
        ax.grid(alpha=.25); ax.set_ylim(0, .9); ax.set_xlim(0, Ls[-1]); ax.set_title(titles[c], loc="left", fontsize=9); ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        if c != 1: ax.text(.98, .95, f"{short(nm3[(ph if c == 0 else pl)[0]])} and {short(nm3[(ph if c == 0 else pl)[1]])}", transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color="#444")
        if r == len(SIZES) - 1: ax.set_xlabel("layer")
    axes[r, 0].set_ylabel("feature sharing (Jaccard)")
axes[-1, 0].legend(fontsize=7, frameon=False, ncol=2, loc="lower left")
fig.tight_layout(); fig.savefig(f"{FIGS}/act3/a3_ladder.png", dpi=140); print("ladder ok")
