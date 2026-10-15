#!/usr/bin/env python3
"""fig_h1_mass.py — thesis figure: where an average prompt's mass goes, by class, at every layer of Gemma 3 27B, for the natural-language,
programming-language and English content domains (three rows, one per group, full width so it reads at page size). Same data and stacking as
v4_story.py H1 (dom_profile_v6_v2.json: machinery first, then each class with its rare features folded in), always-on features removed and the five classes renormalised to the rest of the mass. Output figures/hinge/h1_mass_v2.png."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../lib"))  # thesis/analysis/lib
from paths import WORK, FIGS
import notitle, json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = WORK; CLS = ["position", "token", "word", "concept", "context"]
COL = {"machinery": "#444444", "position": "#8c6d31", "token": "#1f77b4", "word": "#e08a1e", "concept": "#d62728", "context": "#e377c2"}
LANGS = ["english", "finnish", "french", "german", "swedish"]; CODES = ["python", "javascript", "cpp", "c"]
DIVL = [f"div_lang_{l}" if l == "swedish" else f"disjoint_lang_{l}" for l in LANGS]
GROUPS = {"natural languages": [f"par_lang_{l}" for l in LANGS] + DIVL,
          "code": [f"par_code_{c}" for c in CODES] + [f"div_code_{c}" for c in CODES] + ["code_sql", "code_assembly"],
          "content": ["fact_common", "fact_science", "reasoning_legal", "reasoning_math", "reasoning_moral", "story_fiction", "abstract_philosophy", "abstract_religion"]}
D = json.load(open(f"{S}/dom_profile_v6_v2.json")); LD = sorted(int(l) for l in D["layers"]); SIX = CLS
def prof(d):
    P = np.array([[D["layers"][str(L)][d]["mass"][2 + j] + D["layers"][str(L)][d]["rare_mass"][j] for j in range(5)] for L in LD]); return P / np.maximum(P.sum(1, keepdims=True), 1e-9)
plt.rcParams.update({"font.size": 11})
fig, axes = plt.subplots(3, 1, figsize=(9, 7.2), sharex=True)
for ax, (g, doms) in zip(axes, GROUPS.items()):
    M = np.mean([prof(d) for d in doms], 0); base = np.zeros(len(LD))
    for j, c in enumerate(SIX): ax.fill_between(LD, base, base + M[:, j], color=COL[c], lw=0, label=c); base += M[:, j]
    ax.set_ylim(0, 1); ax.set_xlim(0, 61); ax.set_title(f"{g} ({len(doms)} domains)", loc="left", fontsize=11); ax.set_ylabel("by mass: share of\na prompt's mass")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.tick_params(labelbottom=True)
axes[-1].set_xlabel("layer")
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(.5, 1.0), fontsize=10)
fig.tight_layout(rect=(0, 0, 1, .965)); fig.savefig(f"{FIGS}/hinge/h1_mass_v2.png", dpi=170); print("saved")
