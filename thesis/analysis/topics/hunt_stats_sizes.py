#!/usr/bin/env python3
"""hunt_stats_sizes.py SIZE — topic-feature statistics from the per-token captures of scripts/capture.py (<bos> skipped), at the eight layers of
each size that sit at the relative depths of the 27B hunt layers (31, 35, ..., 59 of 62). For every feature with a content home (one
content domain holds >= .5 of the prompts it fires in), 8 to 300 home prompts and corpus rate <= .05: counts per domain, prompts per unit
(FLORES five = one unit, Rosetta four = one unit), mean fired tokens per home prompt and mean coverage (fired tokens / prompt tokens) over
home prompts. No classifier. Output work/hunt_stats3_{size}_L{L}.json. Domains = the 28 of the thesis."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../lib"))  # thesis/analysis/lib
from paths import WORK, capture_dirs
import sys, json, glob, time, collections, numpy as np
SIZE = sys.argv[1]
LAYERS = {"270m": [9, 10, 11, 12, 13, 14, 15, 16], "1b": [13, 14, 16, 18, 19, 21, 23, 24], "4b": [17, 19, 21, 23, 25, 28, 30, 32], "12b": [24, 27, 30, 33, 36, 39, 42, 45], "27b": [31, 35, 39, 43, 47, 51, 55, 59]}
OUT = WORK
RATE_MAX, HOME_MIN, HOME_MAX, DOM, D = 0.05, 8, 300, 0.5, 262144
# the thesis's 28 domains; DROP masks the four earlier diverse-language domains out of every statistic if a capture holds them
DROP = {"div_lang_english", "div_lang_finnish", "div_lang_french", "div_lang_german"}
SRC = [(R, np.load(f"{R}/tokens.npz")) for R in capture_dirs(SIZE)]; R = SRC[0][0]   # one or more capture directories, read as one prompt list
ROWBASE = [int(x) for x in np.cumsum([0] + [int(t["n_tokens"].sum()) for _, t in SRC])[:-1]]
ntok = np.concatenate([t["n_tokens"].astype(np.int64) for _, t in SRC]); dom = np.concatenate([t["domains"] for _, t in SRC]); NP = len(ntok); off = np.concatenate([[0], np.cumsum(ntok)])
keep_p = np.array([d not in DROP for d in dom])
print(SIZE, "captures", [r for r, _ in SRC], "prompts kept", int(keep_p.sum()), flush=True)
DOMS = sorted(set(dom.tolist()) - DROP); CONTENT = [d for d in DOMS if not (d.startswith("par_") or d.startswith("div_") or d.startswith("code_"))]
unit = lambda d: "par_lang" if d.startswith("par_lang_") else ("par_code" if d.startswith("par_code_") else d); UNITS = sorted(set(unit(d) for d in DOMS))
dix = {d: i for i, d in enumerate(DOMS)}; uix = {u: i for i, u in enumerate(UNITS)}; pdom = np.array([dix.get(d, 0) for d in dom]); punit = np.array([uix.get(unit(d), 0) for d in dom]); cidx = [dix[c] for c in CONTENT]
print(SIZE, "prompts", NP, "domains", len(DOMS), "content", CONTENT, flush=True)
for L in LAYERS[SIZE]:
    t0 = time.time(); keys = []
    for (Rx, _), base in zip(SRC, ROWBASE):
        for f in sorted(glob.glob(f"{Rx}/layer_{L}/part_*.npz")):
            z = np.load(f); p = np.searchsorted(off, z["rows"].astype(np.int64) + base, side="right") - 1; m = keep_p[p]; keys.append(z["feats"][m].astype(np.int64) * NP + p[m])
    keys = np.concatenate(keys); k, cnt = np.unique(keys, return_counts=True); feat = (k // NP).astype(np.int64); pr = (k % NP).astype(np.int64); del keys
    C = np.zeros((D, len(DOMS)), np.int32); np.add.at(C, (feat, pdom[pr]), 1)
    tot = C.sum(1); Cc = C[:, cidx]; home_i = Cc.argmax(1); home_n = Cc.max(1); rate = tot / int(keep_p.sum())
    ok = (tot > 0) & (home_n / np.maximum(tot, 1) >= DOM) & (home_n >= HOME_MIN) & (home_n <= HOME_MAX) & (rate <= RATE_MAX)
    want = np.where(ok)[0]; wset = np.zeros(D, bool); wset[want] = True; m = wset[feat]; feat, pr, cnt = feat[m], pr[m], cnt[m]
    home_dom = np.array([dix[CONTENT[i]] for i in home_i]); athome = pdom[pr] == home_dom[feat]
    nf_home = np.zeros(D); cov_home = np.zeros(D); np.add.at(nf_home, feat[athome], cnt[athome]); np.add.at(cov_home, feat[athome], cnt[athome] / ntok[pr[athome]])
    U = np.zeros((D, len(UNITS)), np.int32); np.add.at(U, (feat, punit[pr]), 1)
    feats = []
    for f in want.tolist():
        hd = CONTENT[home_i[f]]; counts = {DOMS[j]: int(C[f, j]) for j in range(len(DOMS)) if C[f, j]}; anyu = {UNITS[j]: int(U[f, j]) for j in range(len(UNITS)) if U[f, j] and UNITS[j] != hd}
        feats.append(dict(f=f, home=hd, home_n=int(home_n[f]), tot=int(tot[f]), mean_nf=float(nf_home[f] / home_n[f]), mean_cov=float(cov_home[f] / home_n[f]), counts=counts, any=anyu))
    out = dict(size=SIZE, layer=L, n_layers=max(int(x.split("_")[-1]) for x in glob.glob(f"{R}/layer_*")) + 1, n_prompts=int(keep_p.sum()), units=UNITS, domains=DOMS, rule=dict(RATE_MAX=RATE_MAX, HOME_MIN=HOME_MIN, HOME_MAX=HOME_MAX, DOM=DOM),
               funnel=dict(all=D, fired=int((tot > 0).sum()), content_home=int(((tot > 0) & (home_n / np.maximum(tot, 1) >= DOM)).sum()), band=int(ok.sum())), features=feats)
    json.dump(out, open(f"{OUT}/hunt_stats3_{SIZE}_L{L}.json", "w")); print(SIZE, L, out["funnel"], round(time.time() - t0), "s", flush=True)
print("STATS3_DONE", SIZE)
