#!/usr/bin/env python3
"""fig_count_mass.py — the count figure and the mass figure of Results 5.2, each one measure, all 378 pairs (left) beside the 16 parallel
pairs (right). f52_count3.png (by count): the share of a domain's set that the other domain's set also holds, stacked by class, for the
nested sets top-1,000, top-5,000 and all fired (bridge_v4_v6_v2/L{L}.json; for all fired the denominator is the domain's own fired set,
the diagonal of 'shared'; both directions averaged; rare features carry their predicted class). f52_mass2.png (by mass): the share of a
domain's mass in features the other domain also fires, by class (ent_mass_cls_27b.json, all fired) and by rank band (ent_mass_262k_27b.json
w_shared, nested top-K differenced into disjoint bands). Machinery removed everywhere; in the count rows it is drawn grey."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../lib"))  # thesis/analysis/lib
from paths import WORK, FIGS
import notitle, json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = WORK; LAY = list(range(62)); LANGS = ["english", "finnish", "french", "german", "swedish"]; CODES = ["python", "javascript", "cpp", "c"]
CLS = ["position", "token", "word", "concept", "context"]; CCOL = {"position": "#8c6d31", "token": "#1f77b4", "word": "#e08a1e", "concept": "#d62728", "context": "#e377c2"}
KCOL = {"top100": "#9ccb3b", "top1000": "#3a9d5d", "top5000": "#1d5c8a", "unbounded": "#b5b5b5"}
D = {L: json.load(open(f"{S}/bridge_v4_v6_v2/L{L}.json")) for L in LAY}; names = D[0]["names"]; ix = {n: i for i, n in enumerate(names)}; n = len(names)
PARN = [(f"par_lang_{a}", f"par_lang_{b}") for i, a in enumerate(LANGS) for b in LANGS[i + 1:]] + [(f"par_code_{a}", f"par_code_{b}") for i, a in enumerate(CODES) for b in CODES[i + 1:]]
PAR = [(ix[a], ix[b]) for a, b in PARN]; ALL = [(a, b) for a in range(n) for b in range(n) if a < b]; PRS = [("all 378 pairs", ALL), ("16 parallel pairs", PAR)]
M = json.load(open(f"{S}/ent_mass_262k_27b.json")); mix = {x: i for i, x in enumerate(M["names"])}; ML = M["layers"]
C = json.load(open(f"{WORK}/ent_mass_cls_27b.json")); cix = {x: i for i, x in enumerate(C["names"])}
def both(A, prs, den=None):   # mean over pairs and both directions of A[a,b]/den[a]
    v = []
    for a, b in prs: v += [A[a, b] / (1 if den is None else den[a]), A[b, a] / (1 if den is None else den[b])]
    return np.mean(v)
plt.rcParams.update({"font.size": 10})
def style(axes):
    for ax in axes.ravel(): ax.set_xlim(0, 61); ax.set_ylim(0, 1.02); ax.grid(alpha=.2); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for ax in axes[-1]: ax.set_xlabel("layer")
rep = []
# ---- count figure
fig, axes = plt.subplots(3, 2, figsize=(12.6, 12.2), sharex=True, sharey=True)
for c, (title, prs) in enumerate(PRS):
    for r, (key, lab) in enumerate([("top1000", "top-1,000"), ("top5000", "top-5,000"), ("unbounded", "all fired")]):
        ax = axes[r, c]; base = np.zeros(62); parts = {}
        for k_ in ["machinery"] + CLS:
            vals = []
            for L in LAY:
                A = np.array(D[L][key][k_], float) + (np.array(D[L][key]["rare_" + k_], float) if k_ != "machinery" else 0)
                den = np.diag(np.array(D[L][key]["shared"], float)); vals.append(both(A, prs, den))
            parts[k_] = np.array(vals)
        for k_, col in [("machinery", "#444444")] + [(cl, CCOL[cl]) for cl in CLS]: ax.fill_between(LAY, base, base + parts[k_], color=col, lw=0, label=("always-on" if k_ == "machinery" else k_)); base += parts[k_]
        tot = np.array([both(np.array(D[L][key]["shared"], float), prs, np.diag(np.array(D[L][key]["shared"], float))) for L in LAY]); ax.plot(LAY, tot, color="k", lw=1.2)
        rep.append(f"count {lab:10s} {title:18s} shared L2 {tot[2]:.2f} L24 {tot[24]:.2f} L61 {tot[61]:.2f} | L24 " + " ".join(f"{cl[:4]} {parts[cl][24]/tot[24]:.2f}" for cl in CLS))
        if c == 0: ax.set_ylabel(f"by count: share of a domain's\n{lab} set that is shared, by class")
    axes[0, c].set_title(title, loc="left", fontsize=11)
style(axes); fig.tight_layout(rect=(0, 0, 1, .96))
h, l = axes[1, 0].get_legend_handles_labels(); fig.legend(h, l, ncol=6, frameon=False, fontsize=10, loc="upper center", bbox_to_anchor=(.5, 1.0))
fig.savefig(f"{FIGS}/act2/f52_count3.png", dpi=170, bbox_inches="tight"); plt.close(fig)
# ---- mass figure
BANDS = [("top100", "the 100 loudest"), ("top1000", "ranks 101 to 1,000"), ("top5000", "ranks 1,001 to 5,000"), ("unbounded", "rank 5,001 to the last")]
mk = {"top100": "100", "top1000": "1000", "top5000": "5000", "unbounded": "all"}
fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.4), sharex=True, sharey=True)
for c, (title, prs) in enumerate(PRS):
    ax = axes[0, c]; base = np.zeros(62)
    for cl in CLS:
        v = np.array([both(np.array(C["layers"][str(L)]["all"][cl], float), [(cix[names[a]], cix[names[b]]) for a, b in prs]) for L in LAY])
        ax.fill_between(LAY, base, base + v, color=CCOL[cl], lw=0, label=cl); base += v
    ax.plot(LAY, base, color="k", lw=1.2); ax.set_title(title, loc="left", fontsize=11); rep.append(f"mass by class {title}: L24 {base[24]:.2f} L61 {base[61]:.2f}")
    ax = axes[1, c]; cum = {key: np.array([both(np.array(ML[str(L)]["w_shared"][mk[key]], float), [(mix[names[a]], mix[names[b]]) for a, b in prs]) for L in LAY]) for key, _ in BANDS}; prev = np.zeros(62)
    for key, lab in BANDS: ax.fill_between(LAY, prev, cum[key], color=KCOL[key], lw=0, label=lab); prev = cum[key]
    ax.plot(LAY, cum["unbounded"], color="k", lw=1.2); rep.append(f"mass by rank {title}: L24 total {cum['unbounded'][24]:.2f}")
axes[0, 0].set_ylabel("by mass: share of a domain's mass\nin shared features, by class"); axes[1, 0].set_ylabel("by mass: share of a domain's mass\nin shared features, by rank band")
style(axes); fig.tight_layout(rect=(0, 0, 1, .93))
h, l = axes[0, 0].get_legend_handles_labels(); fig.legend(h, l, ncol=5, frameon=False, fontsize=10, loc="upper center", bbox_to_anchor=(.5, 1.0))
h, l = axes[1, 0].get_legend_handles_labels(); fig.legend(h, l, ncol=4, frameon=False, fontsize=10, loc="upper center", bbox_to_anchor=(.5, .965))
fig.savefig(f"{FIGS}/act2/f52_mass2.png", dpi=170, bbox_inches="tight"); plt.close(fig)
print("\n".join(rep))
