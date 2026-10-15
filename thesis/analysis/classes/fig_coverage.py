#!/usr/bin/env python3
"""fig_coverage.py — Methods figure: judged cards per layer (Sonnet set, Qwen set) against the features that fire per layer, and the
classifier's out-of-fold agreement with the judges per layer group. Inputs work/tf_train.json, bridge_v4_v6_v2/L*.json."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../lib"))  # thesis/analysis/lib
from paths import WORK, FIGS
import json, re, collections, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier; from sklearn.model_selection import StratifiedKFold, cross_val_predict
S = WORK; d = json.load(open(f"{S}/tf_train.json")); rows = d["rows"]; stats = d["stats"]; CLS = ["position", "token", "word", "concept", "context"]
L = np.array([int(re.search(r"_L(\d+)_", r["id"]).group(1)) for r in rows]); src = np.array([r["src"] for r in rows])
X = np.array([[r["x"].get(s, np.nan) for s in stats] for r in rows], float); y = np.array([CLS.index(r["y"]) for r in rows])
p = cross_val_predict(HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, random_state=0), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
fired = [json.load(open(f"{S}/bridge_v4_v6_v2/L{l}.json"))["n_all"] for l in range(62)]
groups = [(0, 7), (8, 21), (22, 28), (29, 45), (46, 61)]
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1.6, 1]})
a = ax[0]; cs = collections.Counter(L[src == "sonnet"]); cq = collections.Counter(L[src == "qwen_dom"])
a.bar(range(62), [cs.get(l, 0) for l in range(62)], color="#9a9a9a", label="Sonnet set")
a.bar(range(62), [cq.get(l, 0) for l in range(62)], bottom=[cs.get(l, 0) for l in range(62)], color="#1d5c8a", label="Qwen set")
a.set_xlabel("layer"); a.set_ylabel("judged cards"); a.legend(frameon=False, loc="upper right")
a2 = a.twinx(); a2.plot(range(62), np.array(fired) / 1000, color="#222222", lw=1.6, ls="--"); a2.set_ylabel("features that fire in the corpus (thousands)", color="#222222"); a2.set_ylim(0, 260)
b = ax[1]; accs = []; ns = []
for g0, g1 in groups: m = (L >= g0) & (L <= g1); accs.append((p[m] == y[m]).mean()); ns.append(m.sum())
b.bar(range(5), accs, color="#1d5c8a"); b.axhline(0.711, color="#222222", ls="--", lw=1.5); b.set_ylim(0, 1); b.set_xticks(range(5)); b.set_xticklabels([f"L{g0}–{g1}" for g0, g1 in groups])
b.set_ylabel("classifier agreement with the judges, out of fold")
for i, (acc, n) in enumerate(zip(accs, ns)): b.text(i, acc + .02, f"n = {n}", ha="center", fontsize=8)
for x in ax: x.spines["top"].set_visible(False)
fig.tight_layout(); fig.savefig(f"{FIGS}/hinge/coverage.png", dpi=150); print("saved", dict(zip([f"{g0}-{g1}" for g0, g1 in groups], np.round(accs, 3))))
