#!/usr/bin/env python3
"""fig_ladder.py — every fired feature of Gemma 3 27B at the eight hunt layers placed in one of eight groups: always-on (corpus rate >= .6),
topic (the rule of Section 5.3), and otherwise the classifier's
class (position, token, word, concept, context; rare features carry their predicted class). Rare features (under 30 prompts) are split by the classifier's class; the common ones into the eight groups; the figure shows the common features only.
Inputs feat_meta_27b.npz (rate), work/top1_hunt_27b.npz (top1), tf_class_v6, topic_rule_sizes.json. Output figures/spill/ladder_27b.png
and the LaTeX rows on stdout."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../lib"))  # thesis/analysis/lib
from paths import WORK, FIGS
import notitle, json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
Z = np.load(f"{WORK}/top1_hunt_27b.npz"); R = np.load(f"{WORK}/feat_meta_27b.npz")["rate"]; LAY = [31, 35, 39, 43, 47, 51, 55, 59]
T = json.load(open(f"{WORK}/topic_rule_sizes.json"))["features"]["27b"]
G = ["always-on", "position", "token", "word", "concept", "context", "topic"]
COL = {"always-on": "#444444", "single-string": "#0b3d6b", "position": "#8c6d31", "token": "#1f77b4", "word": "#e08a1e", "concept": "#d62728", "context": "#e377c2", "topic": "#7b1fa2"}
rows = {g: [] for g in G}; fired = []; rare = []; rrows = {c: [] for c in ["position", "token", "word", "concept", "context"]}
for L in LAY:
    cls = np.load(f"{WORK}/tf_class_v6/L{L}.npz")["cls"]; top1 = Z[f"top1_{L}"]; rate = R[L]; f = (rate > 0) & (cls < 5); common = f & (rate * 13986 >= 30)
    ao = f & (rate >= .6); ss = np.zeros(262144, bool); tp = np.zeros(262144, bool); tp[[x["f"] for x in T if x["layer"] == L]] = True; tp &= common & ~ao & ~ss; mid = f & ~ao & ~ss & ~tp
    rr = f & ~common; mid = mid & common
    g = {"always-on": ao.sum(), "position": (mid & (cls == 0)).sum(), "token": (mid & (cls == 1)).sum(), "word": (mid & (cls == 2)).sum(), "concept": (mid & (cls == 3)).sum(), "context": (mid & (cls == 4)).sum(), "topic": tp.sum()}
    for k in G: rows[k].append(int(g[k]))
    for i, c in enumerate(rrows): rrows[c].append(int((rr & (cls == i)).sum()))
    fired.append(int(f.sum())); rare.append(int(rr.sum()))
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2)); x = np.arange(len(LAY))
for ax, norm in zip(axes, (False, True)):
    base = np.zeros(len(LAY)); tot = np.array(fired, float) - np.array(rare, float)
    for g in G:
        v = np.array(rows[g], float) / (tot if norm else 1) / (1 if norm else 1000); ax.bar(x, v, bottom=base, color=COL[g], width=.7, label=g); base += v
    ax.set_xticks(x); ax.set_xticklabels([str(L) for L in LAY]); ax.set_xlabel("layer"); ax.set_ylabel("share of the common features" if norm else "common features (thousands)")
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=9, loc="upper center", ncol=7, bbox_to_anchor=(.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, .94)); fig.savefig(f"{FIGS}/spill/ladder_27b.png", dpi=170)
print("Fire in the prompts & " + " & ".join(f"{v:,}" for v in fired) + " \\\\")
print("Rare, under 30 prompts & " + " & ".join(f"{v:,}" for v in rare) + " \\\\")
for c in rrows: print(f"\\quad {c} & " + " & ".join(f"{v:,}" for v in rrows[c]) + " \\\\")
print("Common, 30 prompts or more & " + " & ".join(f"{a - b:,}" for a, b in zip(fired, rare)) + " \\\\")
for g in G: print(f"\\quad {g} & " + " & ".join(f"{v:,}" for v in rows[g]) + " \\\\")
