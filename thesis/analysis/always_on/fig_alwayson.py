#!/usr/bin/env python3
"""fig_alwayson.py — the always-on features of Gemma 3 27B at every layer, for three cuts on the corpus firing rate (the share of the
13,986 prompts a feature fires in): at least .5, .6 (the cut used) and .8. Left: how many features pass each cut (log scale; work/
feat_meta_27b.npz). Right: the share of an average prompt's mass they carry (work/rate_mass_27b.json, from rate_mass_table.py).
Output figures/hinge/alwayson_27b.png."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../lib"))  # thesis/analysis/lib
from paths import WORK, FIGS
import notitle, json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = np.load(f"{WORK}/feat_meta_27b.npz")["rate"]; M = json.load(open(f"{WORK}/rate_mass_27b.json")); LAY = list(range(62))
CUTS = [(.5, "#9ccb3b", "fires in at least 50 % of the prompts"), (.6, "#1d5c8a", "at least 60 % (the cut used)"), (.8, "#222222", "at least 80 %")]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8))
for cut, col, lab in CUTS:
    a.plot(LAY, [(R[L] >= cut).sum() for L in LAY], color=col, lw=2, label=lab)
    b.plot(LAY, [sum(x["mass_share"] for x in M[str(L)]["bands"] if x["lo"] >= cut - 1e-9) for L in LAY], color=col, lw=2, label=lab)
a.set_ylim(0, 130); a.set_ylabel("always-on features at the layer"); b.set_ylim(0, 1); b.set_ylabel("share of a prompt's mass they carry")
for ax in (a, b): ax.set_xlim(0, 61); ax.set_xlabel("layer"); ax.grid(alpha=.25)
a.legend(frameon=False, fontsize=9, loc="upper right"); fig.tight_layout(); fig.savefig(f"{FIGS}/hinge/alwayson_27b.png", dpi=170); print("alwayson ok")
