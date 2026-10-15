#!/usr/bin/env python3
"""fig_52.py — the Section 5.2 figures, one claim each, at most two panels. Data: bridge_v4_v6_v2 (sharing, classes),
ent_mass_262k_27b.json (weight), ent_mass_16k_27b.json (16k). Output figures/act2/f52_{sharing,weight,classes,16k}.png."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../lib"))  # thesis/analysis/lib
from paths import WORK, FIGS
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = WORK; OUT = f"{FIGS}/act2"; B = f"{S}/bridge_v4_v6_v2"; BAND = (22, 29); LAY = list(range(62))
KEYS = ["top100", "top1000", "top5000", "unbounded"]; KL = {"top100": "top-100", "top1000": "top-1,000", "top5000": "top-5,000", "unbounded": "all fired"}
KCOL = {"top100": "#9ccb3b", "top1000": "#3a9d5d", "top5000": "#1d5c8a", "unbounded": "#222222"}; KLS = {"unbounded": "--"}
CLS = ["position", "token", "word", "concept", "context"]; CCOL = {"position": "#8c6d31", "token": "#1f77b4", "word": "#e08a1e", "concept": "#d62728", "context": "#e377c2"}
LANGS = ["english", "finnish", "french", "german", "swedish"]; CODES = ["python", "javascript", "cpp", "c"]
D = {L: json.load(open(f"{B}/L{L}.json")) for L in LAY}; names = D[0]["names"]; ix = {n: i for i, n in enumerate(names)}; n = len(names); iu = np.triu_indices(n, 1)
PAR = [(ix[f"par_lang_{a}"], ix[f"par_lang_{b}"]) for i, a in enumerate(LANGS) for b in LANGS[i + 1:]] + [(ix[f"par_code_{a}"], ix[f"par_code_{b}"]) for i, a in enumerate(CODES) for b in CODES[i + 1:]]
def mat(L, key, k): return np.array(D[L][key][k], float)
def J(L, key):
    sh = mat(L, key, "shared") - mat(L, key, "machinery"); un = mat(L, key, "union") - mat(L, key, "mach_union"); M = sh / np.maximum(un, 1); np.fill_diagonal(M, np.nan); return M
def band(ax): ax.grid(alpha=.25); ax.set_xlim(0, 61); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
def style(key): return dict(color=KCOL[key], lw=2.2, ls=KLS.get(key, "-"), label=KL[key])
rep = []
# 1 sharing: two panels
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, (title, sel) in zip(axes, [("all 378 pairs", lambda M: np.nanmean(M[iu])), ("16 parallel pairs", lambda M: np.nanmean([M[a, b] for a, b in PAR]))]):
    for key in KEYS:
        v = [sel(J(L, key)) for L in LAY]; ax.plot(LAY, v, **style(key)); rep.append(f"sharing {title} {KL[key]} L24 {v[24]:.3f} peak {max(v):.3f} at L{int(np.argmax(v))}")
    band(ax); ax.set_ylim(0, .85); ax.set_title(title, loc="left", fontsize=11); ax.set_xlabel("layer")
axes[0].set_ylabel("feature sharing (Jaccard)"); axes[1].legend(frameon=False, fontsize=9, loc="upper left")
fig.tight_layout(); fig.savefig(f"{OUT}/f52_sharing.png", dpi=160); plt.close(fig)
# 2 weight: one panel, parallel pairs
M = json.load(open(f"{S}/ent_mass_262k_27b.json"))["layers"]; mk = {"top100": "100", "top1000": "1000", "top5000": "5000", "unbounded": "all"}
fig, ax = plt.subplots(figsize=(6.2, 3.8))
cum = {key: np.array([np.mean([np.array(M[str(L)]["w_shared"][mk[key]])[a, b] for a, b in PAR]) for L in LAY]) for key in KEYS}
BANDS = [("top100", "the 100 loudest"), ("top1000", "ranks 101 to 1,000"), ("top5000", "ranks 1,001 to 5,000"), ("unbounded", "rank 5,001 to the last")]
prev = np.zeros(62)
for key, lab in BANDS:
    v = cum[key] - prev; ax.plot(LAY, v, color=KCOL[key], lw=2.2, ls=KLS.get(key, "-"), label=lab); prev = cum[key]
    rep.append(f"weight band {lab} L2 {v[2]:.3f} L24 {v[24]:.3f} L61 {v[61]:.3f}")
band(ax); ax.set_ylim(0, 1.02); ax.set_xlabel("layer"); ax.set_ylabel("share of the domain's mass in the shared features"); ax.legend(frameon=False, fontsize=9, loc="upper right")
fig.tight_layout(); fig.savefig(f"{OUT}/f52_weight.png", dpi=160); plt.close(fig)
# 3 classes: one panel, top-5,000 shared features of the parallel pairs
fig, ax = plt.subplots(figsize=(6.2, 3.8)); key = "top1000"
tot = np.maximum(np.array([np.mean([mat(L, key, "shared")[a, b] for a, b in PAR]) for L in LAY]), 1); base = np.zeros(62)
parts = {"machinery": np.array([np.mean([mat(L, key, "machinery")[a, b] for a, b in PAR]) for L in LAY])}
for cl in CLS: parts[cl] = np.array([np.mean([mat(L, key, cl)[a, b] + mat(L, key, "rare_" + cl)[a, b] for a, b in PAR]) for L in LAY])
for k_, col in [("machinery", "#9a9a9a")] + [(cl, CCOL[cl]) for cl in CLS]:
    v = parts[k_] / tot; ax.fill_between(LAY, base, base + v, color=col, lw=0, label=k_); base += v
band(ax); ax.set_ylim(0, 1); ax.set_xlabel("layer"); ax.set_ylabel("share of the shared top-1,000 features"); ax.legend(frameon=False, fontsize=8.5, loc="upper right", ncol=3)
rep.append("classes top-1,000 par L2 " + " ".join(f"{k_[:4]} {parts[k_][2] / tot[2]:.2f}" for k_ in CLS) + " | L24 " + " ".join(f"{k_[:4]} {parts[k_][24] / tot[24]:.2f}" for k_ in CLS) + " | L61 " + " ".join(f"{k_[:4]} {parts[k_][61] / tot[61]:.2f}" for k_ in CLS))
fig.tight_layout(); fig.savefig(f"{OUT}/f52_classes.png", dpi=160); plt.close(fig)
# 4 16k: one panel, parallel pairs, absolute counts
E = json.load(open(f"{S}/ent_mass_16k_27b.json")); EL = E["layers"]; ix16 = {x: i for i, x in enumerate(E["names"])}; PAR16 = [(ix16[names[a]], ix16[names[b]]) for a, b in PAR]
fig, ax = plt.subplots(figsize=(6.2, 3.8))
for key, m16 in [("top100", "100"), ("top1000", "1000"), ("unbounded", "all")]:
    v = [np.mean([np.array(EL[str(L)]["J"][m16])[a, b] for a, b in PAR16]) for L in LAY]; ax.plot(LAY, v, **style(key)); rep.append(f"16k par {KL[key]} L24 {v[24]:.3f} L0 {v[0]:.3f} peak {max(v):.3f} at L{int(np.argmax(v))}")
band(ax); ax.set_ylim(0, .85); ax.set_xlabel("layer"); ax.set_ylabel("feature sharing (Jaccard), 16k SAE"); ax.legend(frameon=False, fontsize=9, loc="upper left")
fig.tight_layout(); fig.savefig(f"{OUT}/f52_16k.png", dpi=160); plt.close(fig)
print("\n".join(rep)); print("saved")
