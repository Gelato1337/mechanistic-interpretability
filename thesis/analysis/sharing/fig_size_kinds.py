#!/usr/bin/env python3
"""fig_size_kinds.py [K] — feature sharing by pair kind at every layer of the five Gemma 3 sizes, full sets, from work/ent_v4_262k_{size}.json
(as v4_story.py a3_cells, which draws top-5,000). Rows: sizes; left: sentences (FLORES pairs), right: coding tasks (Rosetta pairs).
Same content: the same texts in two languages. Same language: the FLORES or Rosetta domain against the Wikipedia or The Stack domain of the
same language. Neither: FLORES or Rosetta in one language against the Wikipedia or Stack domain of another. Mean and range over the pairs.
x axis: layer, ticks on every panel, panel titles "kind · size" (uniform with a3_sharing / a3_cells). Output figures/act3/a3_kinds_{K}.png."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../lib"))  # thesis/analysis/lib
from paths import WORK, FIGS
import notitle, sys, json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.ticker
K = sys.argv[1] if len(sys.argv) > 1 else "top1000"; S = WORK
LANGS = ["english", "finnish", "french", "german", "swedish"]; CODES = ["python", "javascript", "cpp", "c"]
DIV = {l: (f"div_lang_{l}" if l == "swedish" else f"disjoint_lang_{l}") for l in LANGS}
CELLS = {"sentences": {"same content": [(f"par_lang_{a}", f"par_lang_{b}") for i, a in enumerate(LANGS) for b in LANGS[i + 1:]],
                       "same language": [(f"par_lang_{l}", DIV[l]) for l in LANGS],
                       "neither": [(f"par_lang_{a}", DIV[b]) for a in LANGS for b in LANGS if a != b]},
         "coding tasks": {"same content": [(f"par_code_{a}", f"par_code_{b}") for i, a in enumerate(CODES) for b in CODES[i + 1:]],
                          "same language": [(f"par_code_{c}", f"div_code_{c}") for c in CODES],
                          "neither": [(f"par_code_{a}", f"div_code_{b}") for a in CODES for b in CODES if a != b]}}
COL = {"same content": "#d62728", "same language": "#1f77b4", "neither": "#7f7f7f"}
SIZES = sys.argv[2].split(",") if len(sys.argv) > 2 else ["270m", "1b", "4b", "12b", "27b"]; SL = {"270m": "270M", "1b": "1B", "4b": "4B", "12b": "12B", "27b": "27B"}
KL = {"top100": "top-100", "top1000": "top-1,000", "top5000": "top-5,000", "unbounded": "all fired"}
fig, axes = plt.subplots(len(SIZES), 2, figsize=(10, 2.7 * len(SIZES)), sharey=True, squeeze=False)
for r, s in enumerate(SIZES):
    E = json.load(open(f"{S}/ent_v4_262k_{s}.json")); nm = E["names"]; ix = {d: i for i, d in enumerate(nm)}; Ls = sorted(int(l) for l in E["layers"]); nl = E["n_layers"]
    J = {L: np.array(E["layers"][str(L)]["J"]["0.6"][K] if isinstance(E["layers"][str(L)]["J"].get("0.6"), dict) else E["layers"][str(L)]["J"][K], float) for L in Ls}
    x = np.array(Ls)
    for c, (kind, sets) in enumerate(CELLS.items()):
        ax = axes[r, c]
        for lab, prs in sets.items():
            V = np.array([[J[L][ix[a], ix[b]] for a, b in prs] for L in Ls])
            ax.plot(x, V.mean(1), color=COL[lab], lw=2, label=f"{lab} ({len(prs)} pairs)"); ax.fill_between(x, V.min(1), V.max(1), color=COL[lab], alpha=.13, lw=0)
        ax.set_ylim(0, .9); ax.set_xlim(0, Ls[-1]); ax.grid(alpha=.25); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True)); ax.set_title(f"{kind} · {SL[s]}", loc="left", fontsize=10)
        if c == 0: ax.set_ylabel(f"shared features, Jaccard\n{KL[K]} by mass")
        if r == len(SIZES) - 1: ax.set_xlabel("layer")
for c in range(2): axes[0, c].legend(fontsize=7.5, frameon=False, loc="upper left")
OUT = sys.argv[3] if len(sys.argv) > 3 else f"{FIGS}/act3/a3_kinds_{K}.png"
fig.tight_layout(); fig.savefig(OUT, dpi=150); print("saved", OUT)
