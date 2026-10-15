#!/usr/bin/env python3
"""fig_size_extra.py — two Section 5.3 figures from stored data. (1) a3_mats.png: the top-5,000 pair matrix of every size at its band
centre (270M at its peak layer), 28 domains in the block order of Figure mat. (2) a3_baselines.png: the half-set content gain of the
sentences on the two baselines (Wikipedia openings vs FLORES sentences of other content, top-5,000) and the language gain, against
relative depth, one panel per size. Inputs work/ent_v4_262k_{size}.json, ent_ctl2_{size}.json, ent_ctl2b_{size}.json."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../lib"))  # thesis/analysis/lib
from paths import WORK, FIGS, DIVERSE_LANG
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = WORK; OUT = f"{FIGS}/act3"; SIZES = ["270m", "1b", "4b", "12b", "27b"]; SL = {"270m": "270M", "1b": "1B", "4b": "4B", "12b": "12B", "27b": "27B"}
CENTRE = {"270m": 8, "1b": 10, "4b": 13, "12b": 16, "27b": 25}
LANGS = ["english", "finnish", "french", "german", "swedish"]; CODES = ["python", "javascript", "cpp", "c"]
DIV = DIVERSE_LANG
ORDER = [f"par_lang_{l}" for l in LANGS] + DIV + [f"par_code_{c}" for c in CODES] + [f"div_code_{c}" for c in CODES] + ["code_sql", "code_assembly"] + \
        ["reasoning_math", "reasoning_moral", "reasoning_legal", "abstract_philosophy", "abstract_religion", "fact_science", "fact_common", "story_fiction"]
BOUNDS = [5, 10, 14, 18, 20]; short = lambda d: d.replace("par_lang_", "par ").replace("disjoint_lang_", "div ").replace("div_lang_", "div ").replace("par_code_", "par ").replace("div_code_", "div ").replace("_", " ")
DEPTHS = [0.10, 0.29, 0.52, 1.0]   # layers 6, 18, 32, 61 of 62 in 27B, matched by relative depth in the other sizes
fig, axes = plt.subplots(5, 4, figsize=(10.2, 13.2), constrained_layout=True)   # rows = sizes, columns = depths (portrait, fills the page)
for c, s in enumerate(SIZES):
    d = json.load(open(f"{S}/ent_v4_262k_{s}.json")); names = d["names"]; nl = d["n_layers"]; oi = [names.index(x) for x in ORDER]
    for r, dep in enumerate(DEPTHS):
        L = int(round(dep * (nl - 1))); ax = axes[c, r]
        M = np.array(d["layers"][str(L)]["J"]["0.6"]["top5000"])[np.ix_(oi, oi)]; np.fill_diagonal(M, np.nan); im = ax.imshow(M, cmap="Blues", vmin=0, vmax=0.8)
        for b in BOUNDS: ax.axhline(b - .5, color="#b45309", lw=.7); ax.axvline(b - .5, color="#b45309", lw=.7)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"L{L}", fontsize=10)
        if r == 0: ax.set_ylabel(SL[s], fontsize=12, fontweight="bold")
        if c == 0: ax.annotate(f"depth {dep:.2f}", xy=(.5, 1.18), xycoords="axes fraction", ha="center", fontsize=11, fontweight="bold")
fig.colorbar(im, ax=axes, shrink=.5, pad=.01, label="Jaccard of the two domains' top-5,000 feature sets")
fig.savefig(f"{OUT}/a3_mats_grid.png", dpi=150, bbox_inches="tight"); plt.close(fig)
fig, axes = plt.subplots(2, 3, figsize=(11, 7), sharey=True); axes = axes.ravel(); axes[-1].axis("off")
for k, s in enumerate(SIZES):
    a = json.load(open(f"{S}/ent_ctl2_{s}.json"))["layers"]; b = json.load(open(f"{S}/ent_ctl2b_{s}.json"))["layers"]; n = len(a); x = np.arange(n) / (n - 1)
    sc = np.array([a[str(l)]["same_source"]["sentences"]["same content"]["top5000"] for l in range(n)])
    sl = np.array([a[str(l)]["same_source"]["sentences"]["same language"]["top5000"] for l in range(n)])
    oo = np.array([a[str(l)]["same_source"]["sentences"]["same source other content"]["top5000"] for l in range(n)])
    wiki = np.array([b[str(l)]["top5000"] for l in range(n)])
    ax = axes[k]; ax.axhline(0, color="#999", lw=.8)
    ax.plot(x, sc - wiki, color="#d62728", lw=2, ls="--", label="content gain, Wikipedia baseline")
    ax.plot(x, sc - oo, color="#d62728", lw=2, label="content gain, same-source baseline")
    ax.plot(x, sl - oo, color="#1f5fa8", lw=2, label="language gain, same-source baseline")
    ax.set_title(SL[s], fontsize=12, loc="left"); ax.set_xlabel("relative depth"); ax.set_ylim(-.1, .55); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
axes[0].set_ylabel("gain in Jaccard, top-5,000, half sets"); axes[3].set_ylabel("gain in Jaccard, top-5,000, half sets"); axes[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(f"{OUT}/a3_baselines.png", dpi=150); plt.close(fig); print("saved")
