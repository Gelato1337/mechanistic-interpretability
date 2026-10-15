#!/usr/bin/env python3
"""fig_size_sharing.py — Figure sharing-by-depth for all five sizes: rows = sizes, columns = all 378 pairs | 16 parallel pairs, four set
sizes (top-100, top-1,000, top-5,000, all fired), Jaccard, machinery removed, lenm_ domains excluded. Data ent_v4_262k_{size}.json.
Also a3_cells_half.png: content vs language on the half sets per size (ent_ctl2_{size}.json, ent_ctl2b_{size}.json), sentences | programs.
Output figures/act3/a3_sharing.png, a3_cells_half.png."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../lib"))  # thesis/analysis/lib
from paths import WORK, FIGS
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = WORK; SIZES = ["270m", "1b", "4b", "12b", "27b"]; SL = {"270m": "270M", "1b": "1B", "4b": "4B", "12b": "12B", "27b": "27B"}
KEYS = ["top100", "top1000", "top5000", "unbounded"]; KL = {"top100": "top-100", "top1000": "top-1,000", "top5000": "top-5,000", "unbounded": "all fired"}
KCOL = {"top100": "#9ccb3b", "top1000": "#3a9d5d", "top5000": "#1d5c8a", "unbounded": "#222222"}; KLS = {"unbounded": "--"}
LANGS = ["english", "finnish", "french", "german", "swedish"]; CODES = ["python", "javascript", "cpp", "c"]
fig, axes = plt.subplots(5, 2, figsize=(11, 15), sharey=True)
for r, s in enumerate(SIZES):
    d = json.load(open(f"{S}/ent_v4_262k_{s}.json")); names = d["names"]; keep = [i for i, nm in enumerate(names) if not nm.startswith("lenm_")]; nm = [names[i] for i in keep]; ix = {x: i for i, x in enumerate(nm)}
    PAR = [(ix[f"par_lang_{a}"], ix[f"par_lang_{b}"]) for i, a in enumerate(LANGS) for b in LANGS[i + 1:]] + [(ix[f"par_code_{a}"], ix[f"par_code_{b}"]) for i, a in enumerate(CODES) for b in CODES[i + 1:]]
    Ls = sorted(int(k) for k in d["layers"]); iu = np.triu_indices(len(keep), 1)
    for c, (title, sel) in enumerate([("all 378 pairs", lambda M: M[iu].mean()), ("16 parallel pairs", lambda M: np.mean([M[a, b] for a, b in PAR]))]):
        ax = axes[r, c]
        for key in KEYS:
            v = [sel(np.array(d["layers"][str(L)]["J"]["0.6"][key])[np.ix_(keep, keep)]) for L in Ls]; ax.plot(Ls, v, color=KCOL[key], lw=2, ls=KLS.get(key, "-"), label=KL[key])
        ax.grid(alpha=.25); ax.set_ylim(0, .9); ax.set_xlim(0, Ls[-1]); ax.set_title(f"{SL[s]} · {title}", loc="left", fontsize=10.5); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        if r == 4: ax.set_xlabel("layer")
    axes[r, 0].set_ylabel("feature sharing (Jaccard)")
axes[0, 1].legend(frameon=False, fontsize=8.5, loc="upper left")
fig.tight_layout(); fig.savefig(f"{FIGS}/act3/a3_sharing.png", dpi=150); plt.close(fig)
fig, axes = plt.subplots(5, 2, figsize=(11, 15), sharey=True)
for r, s in enumerate(SIZES):
    a = json.load(open(f"{S}/ent_ctl2_{s}.json"))["layers"]; b = json.load(open(f"{S}/ent_ctl2b_{s}.json"))["layers"]; Ls = sorted(int(k) for k in a)
    g = lambda grp, k: [a[str(l)]["same_source"][grp][k]["top5000"] for l in Ls]
    ax = axes[r, 0]
    ax.plot(Ls, g("sentences", "same content"), color="#d62728", lw=2, label="same sentences, other language"); ax.plot(Ls, g("sentences", "same language"), color="#1f5fa8", lw=2, label="other sentences, same language")
    ax.plot(Ls, g("sentences", "same source other content"), color="#7f7f7f", lw=2, label="other sentences, other language"); ax.plot(Ls, [b[str(l)]["top5000"] for l in Ls], color="#7f7f7f", lw=1.5, ls="--", label="Wikipedia text, other language")
    ax.set_title(f"{SL[s]} · sentences", loc="left", fontsize=10.5); ax.set_ylabel("Jaccard, top-5,000, half sets")
    ax = axes[r, 1]
    ax.plot(Ls, g("programs", "same content"), color="#d62728", lw=2, label="same task, other language"); ax.plot(Ls, g("programs", "same language"), color="#1f5fa8", lw=2, label="other tasks, same language"); ax.plot(Ls, g("programs", "same source other content"), color="#7f7f7f", lw=2, label="other tasks, other language")
    ax.set_title(f"{SL[s]} · programs", loc="left", fontsize=10.5)
    for ax in axes[r]: ax.grid(alpha=.25); ax.set_ylim(0, .9); ax.set_xlim(0, Ls[-1]); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    if r == 4:
        for ax in axes[r]: ax.set_xlabel("layer")
axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left"); axes[0, 1].legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(f"{FIGS}/act3/a3_cells_half.png", dpi=150); print("saved")
