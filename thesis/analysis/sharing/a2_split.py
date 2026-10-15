#!/usr/bin/env python3
"""a2_split.py — all-fired feature sharing split into features seen >= 30 times vs < 30 times in the corpus, every layer of 27B,
from work/bridge_v4_v6_v2/L*.json (unbounded: union, shared, rare = shared rare; diagonal of rare = the domain's rare count).
Writes work/split_common_rare_v2.json and figures/act2/a2_split.png (thesis Figure split)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../lib"))  # thesis/analysis/lib
from paths import WORK, FIGS
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = WORK; Ls = list(range(62)); rows = []
for L in Ls:
    u = json.load(open(f"{S}/bridge_v4_v6_v2/L{L}.json"))["unbounded"]
    U = np.array(u["union"], float); Sh = np.array(u["shared"], float); R = np.array(u["rare"], float)
    n = U.shape[0]; iu = np.triu_indices(n, 1); setsize = np.diag(U); rare_d = np.diag(R)
    RU = rare_d[:, None] + rare_d[None, :] - R; Uc = U - RU; Sc = Sh - R
    rows.append(dict(L=L, J=(Sh / U)[iu].mean(), Jc=(Sc / Uc)[iu].mean(), Jr=(R / RU)[iu].mean(), frac=(RU / U)[iu].mean(), common=(setsize - rare_d).mean(), rare=rare_d.mean()))
json.dump(rows, open(f"{S}/split_common_rare_v2.json", "w"))
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6)); a, b = ax
a.plot(Ls, [r["Jc"] for r in rows], color="#1f5fa8", lw=2, label="seen ≥ 30 times"); a.plot(Ls, [r["J"] for r in rows], color="#444", lw=2, ls="--", label="all fired"); a.plot(Ls, [r["Jr"] for r in rows], color="#c98f1a", lw=2, label="seen < 30 times")
a.set_xlabel("layer"); a.set_ylabel("Jaccard, mean over 378 pairs"); a.set_ylim(0, 0.8); a.legend(frameon=False, loc="upper left")
b.stackplot(Ls, [r["common"] for r in rows], [r["rare"] for r in rows], colors=["#1f5fa8", "#c98f1a"], labels=["seen ≥ 30 times", "seen < 30 times"], alpha=.9)
b.set_xlabel("layer"); b.set_ylabel("fired features per domain"); b.legend(frameon=False, loc="upper right")
for x in ax: x.spines["top"].set_visible(False); x.spines["right"].set_visible(False)
plt.tight_layout(); plt.savefig(f"{FIGS}/act2/a2_split.png", dpi=200); print("saved")
