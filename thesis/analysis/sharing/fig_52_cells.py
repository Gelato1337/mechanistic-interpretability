#!/usr/bin/env python3
"""fig_52_cells.py — content against language on the stored half sets, top-5,000 by mass, 27B. Sentences: the same sentences in another
language / other FLORES sentences in the same language / other FLORES sentences in another language / Wikipedia openings in another
language. Programs: the same task in another language / other tasks in the same language / other tasks in another language.
Data ent_ctl2_27b.json, ent_ctl2b_27b.json. Output figures/act2/f52_cells.png."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../lib"))  # thesis/analysis/lib
from paths import WORK, FIGS
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = WORK; a = json.load(open(f"{S}/ent_ctl2_27b.json"))["layers"]; b = json.load(open(f"{S}/ent_ctl2b_27b.json"))["layers"]; L = list(range(62)); BAND = (22, 29)
g = lambda grp, k: [a[str(l)]["same_source"][grp][k]["top5000"] for l in L]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
ax = axes[0]
ax.plot(L, g("sentences", "same content"), color="#d62728", lw=2.2, label="same sentences, other language")
ax.plot(L, g("sentences", "same language"), color="#1f5fa8", lw=2.2, label="other sentences, same language")
ax.plot(L, g("sentences", "same source other content"), color="#7f7f7f", lw=2.2, label="other sentences, other language")
ax.plot(L, [b[str(l)]["top5000"] for l in L], color="#7f7f7f", lw=1.6, ls="--", label="Wikipedia text, other language")
ax.set_title("sentences", loc="left", fontsize=11); ax.set_ylabel("feature sharing (Jaccard), top-5,000, half sets"); ax.legend(frameon=False, fontsize=8.5, loc="upper left")
ax = axes[1]
ax.plot(L, g("programs", "same content"), color="#d62728", lw=2.2, label="same task, other language")
ax.plot(L, g("programs", "same language"), color="#1f5fa8", lw=2.2, label="other tasks, same language")
ax.plot(L, g("programs", "same source other content"), color="#7f7f7f", lw=2.2, label="other tasks, other language")
ax.set_title("coding tasks", loc="left", fontsize=11); ax.legend(frameon=False, fontsize=8.5, loc="lower left")
for ax in axes: ax.grid(alpha=.25); ax.set_xlim(0, 61); ax.set_ylim(0, .85); ax.set_xlabel("layer"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.tight_layout(); fig.savefig(f"{FIGS}/act2/f52_cells.png", dpi=160)
for l in (2, 22, 24, 33, 61): print(l, {k: round(v, 2) for k, v in zip(["sc", "sl", "oo", "wiki"], [g("sentences", "same content")[l], g("sentences", "same language")[l], g("sentences", "same source other content")[l], b[str(l)]["top5000"]])}, "prog", {k: round(g("programs", kk)[l], 2) for k, kk in [("sc", "same content"), ("sl", "same language"), ("oo", "same source other content")]})
