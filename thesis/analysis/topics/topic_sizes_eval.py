#!/usr/bin/env python3
"""topic_sizes_eval.py — the topic-feature rule on the five sizes (work/hunt_stats3_{size}_L{L}.json, thesis 28 domains):
one content home with 25 to 300 home prompts and rate <= .05; at least 4 fired tokens per home prompt and coverage >= 3 % of the home
prompts' tokens; fires in >= 2 prompts of >= 5 other domains. Prints the funnel per size and layer, the 27B agreement with the classifier
(tf_class_v6) and the gallery survival, writes work/topic_rule_sizes.json (funnel + per-feature records of the topic features)
and draws figures/spill/topic_matrix_sizes.png (one matrix per size, pooled over its eight layers) and topic_matrix_all.png (27B)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../lib"))  # thesis/analysis/lib
from paths import WORK, FIGS
import json, re, collections, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import notitle
SIZES = {"270m": [9, 10, 11, 12, 13, 14, 15, 16], "1b": [13, 14, 16, 18, 19, 21, 23, 24], "4b": [17, 19, 21, 23, 25, 28, 30, 32], "12b": [24, 27, 30, 33, 36, 39, 42, 45], "27b": [31, 35, 39, 43, 47, 51, 55, 59]}
SL = {"270m": "270M", "1b": "1B", "4b": "4B", "12b": "12B", "27b": "27B"}; NF, COV, UNITS_MIN, HMIN, HMAX = 4, .03, 5, 25, 300
reach = lambda x: sorted(u for u, n in x["any"].items() if n >= 2); rule = lambda x: HMIN <= x["home_n"] <= HMAX and x["mean_nf"] >= NF and x["mean_cov"] >= COV and len(reach(x)) >= UNITS_MIN
import os   # optional: the earlier hand-reviewed gallery page, to report how many of its features the rule keeps
h = open(f"{WORK}/spill_pages/spill_hunt.html").read() if os.path.exists(f"{WORK}/spill_pages/spill_hunt.html") else ""; G = collections.defaultdict(set)
for L, f in set(re.findall(r"L(\d+)[^<]{0,40}?f(\d+)", h)): G[int(L)].add(int(f))
out = {"rule": dict(home_min=HMIN, home_max=HMAX, rate_max=.05, home_share=.5, fired_tokens_per_home_prompt=NF, coverage=COV, other_domains=UNITS_MIN), "sizes": {}}; topic = {}
for s, Ls in SIZES.items():
    out["sizes"][s] = {"layers": {}}; topic[s] = []
    for L in Ls:
        d = json.load(open(f"{WORK}/hunt_stats3_{s}_L{L}.json")); F = d["features"]; units = d["units"]
        r1 = [x for x in F if HMIN <= x["home_n"] <= HMAX]; r2 = [x for x in r1 if x["mean_nf"] >= NF and x["mean_cov"] >= COV]; r3 = [x for x in r2 if len(reach(x)) >= UNITS_MIN]
        rec = dict(n_layers=d["n_layers"], all=d["funnel"]["all"], fired=d["funnel"]["fired"], home=len(r1), spans=len(r2), topic=len(r3))
        if s == "27b":
            cls = np.load(f"{WORK}/tf_class_v6/L{L}.npz")["cls"]; pool = [x for x in r1 if len(reach(x)) >= UNITS_MIN]; y = np.array([cls[x["f"]] == 4 for x in pool]); p = np.array([rule(x) for x in pool]); tp = int((p & y).sum())
            rec.update(classifier_context_in_pool=int(y.sum()), precision=round(tp / max(p.sum(), 1), 3), recall=round(tp / max(y.sum(), 1), 3), gallery_kept=sum(x["f"] in G[L] for x in r3), gallery_total=len(G[L]), gallery_dropped=sorted(f for f in G[L] if f not in {x["f"] for x in r3}))
        out["sizes"][s]["layers"][str(L)] = rec; topic[s] += [dict(x, layer=L, reached=reach(x)) for x in r3]
    out["sizes"][s]["units"] = units; out["sizes"][s]["topic_total"] = len(topic[s])
json.dump({**out, "features": {s: topic[s] for s in SIZES}}, open(f"{WORK}/topic_rule_sizes.json", "w"))
print(f"{'size':5s} {'layer':5s} {'fired':>8s} {'home':>6s} {'spans':>6s} {'topic':>6s}   27B: precision recall gallery")
for s in SIZES:
    for L, r in out["sizes"][s]["layers"].items(): print(f"{SL[s]:5s} {L:5s} {r['fired']:8,d} {r['home']:6d} {r['spans']:6d} {r['topic']:6d}" + (f"   {r['precision']:.2f} {r['recall']:.2f} {r['gallery_kept']}/{r['gallery_total']} dropped {r['gallery_dropped']}" if s == "27b" else ""))
    print(f"{SL[s]} total topic features {out['sizes'][s]['topic_total']}")
# matrices
def matrix(ax, feats, units, title):
    CONTENT = [u for u in units if not (u.startswith("par_") or u.startswith("div_") or u.startswith("code_") or u.startswith("disjoint_"))]; U = CONTENT + [u for u in units if u.startswith(("par_code", "div_code", "code_"))] + [u for u in units if u.startswith(("par_lang", "disjoint_lang", "div_lang"))]
    M = np.full((len(CONTENT), len(U)), np.nan); N = []
    for i, hm in enumerate(CONTENT):
        F = [x for x in feats if x["home"] == hm]; N.append(len(F))
        for j, u in enumerate(U):
            if u != hm: M[i, j] = 100 * sum(1 for x in F if x["any"].get(u, 0) >= 2) / max(len(F), 1)
    im = ax.imshow(np.ma.masked_invalid(M), cmap="Blues", vmin=0, vmax=100, aspect="auto"); ax.set_xticks(range(len(U))); ax.set_xticklabels(U, rotation=60, ha="right", fontsize=7); ax.set_yticks(range(len(CONTENT))); ax.set_yticklabels([f"{hm}  (n={n})" for hm, n in zip(CONTENT, N)], fontsize=7)
    for i in range(len(CONTENT)):
        for j in range(len(U)):
            if not np.isnan(M[i, j]): ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=6, color="white" if M[i, j] > 50 else "black")
    ncode = len([u for u in U if u.startswith(("par_code", "div_code", "code_"))]); ax.axvline(len(CONTENT) - .5, color="k", lw=1); ax.axvline(len(CONTENT) + ncode - .5, color="k", lw=1); ax.set_title(title, loc="left", fontsize=10); return im
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
for ax, s in zip(axes.ravel(), [k for k in SIZES if k != "27b"]): im = matrix(ax, topic[s], out["sizes"][s]["units"], f"{SL[s]} · {len(topic[s]):,} topic features")
fig.tight_layout(); fig.savefig(f"{FIGS}/spill/topic_matrix_sizes.png", dpi=150); plt.close(fig)
fig, ax = plt.subplots(figsize=(12, 7.6)); plt.rcParams.update({"font.size": 11}); im = matrix(ax, topic["27b"], out["sizes"]["27b"]["units"], ""); ax.set_title("")
for t in ax.get_xticklabels(): t.set_fontsize(11)
for t in ax.get_yticklabels(): t.set_fontsize(11)
for t in ax.texts: t.set_fontsize(10)
cb = fig.colorbar(im, ax=ax, fraction=.025, pad=.01, ticks=[0, 25, 50, 75, 100]); cb.set_label("share of the home's topic features (%)", fontsize=10)
fig.tight_layout(); fig.savefig(f"{FIGS}/spill/topic_matrix_all.png", dpi=160); print("figures ok")
