#!/usr/bin/env python3
"""hunt_spill.py L — one-pass hunt for context-like spill features at layer L of the 27B capture.
A candidate has a home content domain holding >= DOM of all prompts it fires in, HOME_MIN..HOME_MAX home prompts, rate <= RATE_MAX,
fires on >= MEAN_NF tokens per home prompt on average (span, not single word), and has >= 1 non-home prompt with >= CTX_NF fired tokens.
Ranked per home by (number of non-home domains with a >= CTX_NF passage, number of such passages). Top N per home.
Output work/hunt_spill_L{L}.json: per feature counts per domain, home passages (3) and spill passages (2 per domain, >= CTX_NF fires), full tokens up to MAXTOK."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../lib"))  # thesis/analysis/lib
from paths import WORK, capture, DIVERSE_LANG
from shards import load_domains
import sys, json, time, collections, numpy as np
S = f"{WORK}"; L = int(sys.argv[1])
N = 8; RATE_MAX = 0.05; HOME_MIN = 8; HOME_MAX = 300; DOM = 0.5; MEAN_NF = 4.0; CTX_NF = 3; D = 262144; MAXTOK = 320; PER_HOME = 3; PER_SPILL = 2
LANGS = ["english", "finnish", "french", "german", "swedish"]; CODES = ["python", "javascript", "cpp", "c"]
DOMS = [f"par_lang_{l}" for l in LANGS] + DIVERSE_LANG + [f"par_code_{c}" for c in CODES] + [f"div_code_{c}" for c in CODES] + \
       ["code_sql", "code_assembly", "reasoning_math", "reasoning_moral", "reasoning_legal", "abstract_philosophy", "abstract_religion", "fact_science", "fact_common", "story_fiction"]
CONTENT = DOMS[20:]
t0 = time.time(); cap = load_domains(capture("27b"), L); print("loaded", round(time.time() - t0), flush=True)
cnt = {d: np.zeros(D, np.int64) for d in DOMS}; nf_home = np.zeros(D); NP = 0
for dom in DOMS:
    dd = cap[dom]
    for j in range(len(dd["token_strings"])):
        pos, fid, act = (np.asarray(x) for x in dd["activations"][j]); m = pos > 0; fid = fid[m].astype(np.int64)
        if len(fid) == 0: continue
        NP += 1; nf = np.bincount(fid, minlength=D); cnt[dom] += nf > 0
        if dom in CONTENT: nf_home += nf   # fired-token count summed over content prompts (home dominates for selected features)
    print(dom, round(time.time() - t0), flush=True)
tot = sum(cnt[d] for d in DOMS); C = np.stack([cnt[d] for d in CONTENT]); home_i = C.argmax(0); home_n = C.max(0); rate = tot / NP
ok = (home_n >= HOME_MIN) & (home_n <= HOME_MAX) & (home_n / np.maximum(tot, 1) >= DOM) & (rate <= RATE_MAX) & (tot > home_n) & (nf_home / np.maximum(home_n, 1) >= MEAN_NF)
want = np.where(ok)[0]; wset = set(want.tolist()); print("candidates", len(want), flush=True)
hits = {int(f): collections.defaultdict(list) for f in want}   # f -> dom -> [(nfire, peak, j, {pos: act})]
for dom in DOMS:
    dd = cap[dom]
    for j in range(len(dd["token_strings"])):
        pos, fid, act = (np.asarray(x) for x in dd["activations"][j]); m = (pos > 0) & np.isin(fid, want)
        if not m.any(): continue
        for f in set(fid[m].tolist()):
            mm = m & (fid == f); hits[int(f)][dom].append((int(mm.sum()), float(act[mm].max()), j, {int(p): float(a) for p, a in zip(pos[mm], act[mm])}))
    print("pass2", dom, round(time.time() - t0), flush=True)
def passage(dom, nfire, peak, j, fires):
    toks = cap[dom]["token_strings"][j][1:]; n = len(toks)
    return dict(prompt=j, peak=peak, n_tokens=n, n_fired=nfire, tokens=toks[:MAXTOK], fires={str(p - 1): a for p, a in fires.items() if p - 1 < MAXTOK})
out = {"layer": L, "n_prompts": int(NP), "homes": {}}
for hi, hd in enumerate(CONTENT):
    rows = []
    for f in want[home_i[want] == hi]:
        f = int(f); h = hits[f]
        ctx = {d: [x for x in v if x[0] >= CTX_NF] for d, v in h.items() if d != hd}; ctx = {d: v for d, v in ctx.items() if v}
        if not ctx: continue
        rows.append((len(ctx), sum(len(v) for v in ctx.values()), f, ctx))
    rows.sort(key=lambda r: (-r[0], -r[1]))
    lst = []
    for nd, npass, f, ctx in rows[:N]:
        h = hits[f]; rec = dict(f=f, home=hd, home_prompts=int(home_n[f]), prompts=int(tot[f]), mean_nf_home=float(nf_home[f] / home_n[f]), counts={d: len(v) for d, v in h.items()}, passages={})
        rec["passages"][hd] = [passage(hd, *x) for x in sorted(h[hd], key=lambda x: -x[0])[:PER_HOME]]
        for d, v in sorted(ctx.items(), key=lambda kv: -len(kv[1])): rec["passages"][d] = [passage(d, *x) for x in sorted(v, key=lambda x: -x[0])[:PER_SPILL]]
        lst.append(rec)
    out["homes"][hd] = lst; print(hd, "kept", len(lst), "of", len(rows), flush=True)
json.dump(out, open(f"{S}/hunt_spill_L{L}.json", "w"), ensure_ascii=False); print("HUNT_DONE", L)
