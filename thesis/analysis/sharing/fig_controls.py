#!/usr/bin/env python3
"""fig_controls.py — the four control figures of Appendix D in one style (same panel size, line width, legend place, axis labels, band
shading): f52_16k.png (second dictionary, 16k SAE, parallel pairs; ent_mass_16k_27b.json), a2_ctl_robust.png (split halves, machinery
threshold, fixed fraction; ent_v4_262k_27b.json), a2_ctl_length.png (length control, sentences, three set sizes; lenm_lang_* domains of
ent_v4_262k_27b.json), a2_ctl_weighted.png (weighted Jaccard of the mass vectors; ent_ctl2_27b.json). 27B, machinery removed."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../lib"))  # thesis/analysis/lib
from paths import WORK, FIGS, UNUSED_DIVERSE
import notitle, json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = WORK; OUT = f"{FIGS}/act2"; LAY = list(range(62)); BAND = (21.5, 29.5)
LANGS = ["english", "finnish", "french", "german", "swedish"]; CODES = ["python", "javascript", "cpp", "c"]
DIV = {l: (f"div_lang_{l}" if l == "swedish" else f"disjoint_lang_{l}") for l in LANGS}
KEYS = ["top1000", "top5000", "unbounded"]; KL = {"top100": "top-100", "top1000": "top-1,000", "top5000": "top-5,000", "unbounded": "all fired"}
KST = {"top100": ("#9ccb3b", "-"), "top1000": ("#3a9d5d", "-"), "top5000": ("#1d5c8a", "-"), "unbounded": ("#222222", "--")}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
def style(ax, ylabel="feature sharing (Jaccard)", ylim=(0, .9)):
    ax.grid(alpha=.25); ax.set_xlim(0, 61); ax.set_ylim(*ylim); ax.set_xlabel("layer"); ax.set_ylabel(ylabel)
E = json.load(open(f"{S}/ent_v4_262k_27b.json")); nm = E["names"]; DROP = UNUSED_DIVERSE
kp = [i for i, d in enumerate(nm) if d not in DROP and not d.startswith("lenm_")]; jn = {d: i for i, d in enumerate(nm)}; iu = np.triu_indices(len(kp), 1)
def JM(L, key, thr="0.6", half=None):
    r = E["layers"][str(L)]
    if half is not None: return np.array(r["J_half"][key][half], float)
    return np.array(r["J"][thr][key] if isinstance(r["J"].get("0.6"), dict) else r["J"][key], float)
def mean_all(key, thr="0.6", half=None): return np.array([np.nanmean(JM(L, key, thr, half)[np.ix_(kp, kp)][iu]) for L in LAY])
# 1 second dictionary
M16 = json.load(open(f"{S}/ent_mass_16k_27b.json")); EL = M16["layers"]; ix16 = {x: i for i, x in enumerate(M16["names"])}
PAR = [(ix16[f"par_lang_{a}"], ix16[f"par_lang_{b}"]) for i, a in enumerate(LANGS) for b in LANGS[i + 1:]] + [(ix16[f"par_code_{a}"], ix16[f"par_code_{b}"]) for i, a in enumerate(CODES) for b in CODES[i + 1:]]
fig, ax = plt.subplots(figsize=(6.5, 3.8))
for key, m16 in [("top100", "100"), ("top1000", "1000"), ("unbounded", "all")]:
    col, ls = KST[key]; ax.plot(LAY, [np.mean([np.array(EL[str(L)]["J"][m16])[a, b] for a, b in PAR]) for L in LAY], color=col, ls=ls, lw=2, label=KL[key])
style(ax, "feature sharing (Jaccard), 16k SAE"); ax.legend(frameon=False, fontsize=9, loc="upper left"); fig.tight_layout(); fig.savefig(f"{OUT}/f52_16k.png", dpi=170); plt.close(fig)
# 2 split halves, machinery threshold, fixed fraction
fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), sharey=True)
for key in KEYS:
    col, ls = KST[key]; axes[0].plot(LAY, mean_all(key, half=0), color=col, ls=ls, lw=2, label=f"{KL[key]}, odd prompts"); axes[0].plot(LAY, mean_all(key, half=1), color=col, ls=ls, lw=2, alpha=.4, label=f"{KL[key]}, even prompts")
    for thr, a in (("0.5", .35), ("0.6", 1), ("0.8", .6)): axes[1].plot(LAY, mean_all(key, thr), color=col, ls=ls, lw=2 if thr == "0.6" else 1.2, alpha=a, label=f"{KL[key]}, always-on cut at {thr}")
axes[2].plot(LAY, mean_all("top5000"), color=KST["top5000"][0], lw=2, label="top-5,000, a fixed count"); axes[2].plot(LAY, mean_all("top10pct"), color="#b45309", lw=2, label="top 10 per cent of what fires"); axes[2].plot(LAY, mean_all("unbounded"), color="#222222", ls="--", lw=2, label="all fired")
for ax, t in zip(axes, ("split halves", "machinery threshold", "fixed fraction")): style(ax); ax.set_title(t, loc="left", fontsize=11); ax.legend(frameon=False, fontsize=7.5, loc="upper left")
for ax in axes[1:]: ax.set_ylabel("")
fig.tight_layout(); fig.savefig(f"{OUT}/a2_ctl_robust.png", dpi=170); plt.close(fig)
# 3 length
LM = {l: f"lenm_lang_{l}" for l in LANGS}
SETS = {"same content": [(f"par_lang_{a}", f"par_lang_{b}") for i, a in enumerate(LANGS) for b in LANGS[i + 1:]], "same language, full length": [(f"par_lang_{l}", DIV[l]) for l in LANGS], "same language, cut to the sentence length": [(f"par_lang_{l}", LM[l]) for l in LANGS],
        "neither, full length": [(f"par_lang_{a}", DIV[b]) for a in LANGS for b in LANGS if a != b], "neither, cut to the sentence length": [(f"par_lang_{a}", LM[b]) for a in LANGS for b in LANGS if a != b]}
SCOL = {"same content": ("#d62728", "-"), "same language, full length": ("#1f77b4", "-"), "same language, cut to the sentence length": ("#1f77b4", "--"), "neither, full length": ("#7f7f7f", "-"), "neither, cut to the sentence length": ("#7f7f7f", "--")}
fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), sharey=True)
for ax, key in zip(axes, KEYS):
    for lab, prs in SETS.items():
        col, ls = SCOL[lab]; V = np.array([[JM(L, key)[jn[a], jn[b]] for a, b in prs] for L in LAY]); ax.plot(LAY, V.mean(1), color=col, ls=ls, lw=2, label=f"{lab} ({len(prs)} pairs)")
    style(ax); ax.set_title(f"sentences · {KL[key]}", loc="left", fontsize=11)
for ax in axes[1:]: ax.set_ylabel("")
axes[0].legend(frameon=False, fontsize=7.5, loc="upper left"); fig.tight_layout(); fig.savefig(f"{OUT}/a2_ctl_length.png", dpi=170); plt.close(fig)
# 4 weighted overlap
d = json.load(open(f"{S}/ent_ctl2_27b.json"))["layers"]; fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.plot(LAY, [d[str(l)]["ruzicka"] for l in LAY], color="#222222", lw=2, label="all 378 pairs"); ax.plot(LAY, [d[str(l)]["ruzicka_par"] for l in LAY], color="#d62728", lw=2, label="16 parallel pairs")
style(ax, "weighted Jaccard of the mass vectors", (0, .8)); ax.legend(frameon=False, fontsize=9, loc="upper left"); fig.tight_layout(); fig.savefig(f"{OUT}/a2_ctl_weighted.png", dpi=170); plt.close(fig)
print("controls ok")
