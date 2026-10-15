#!/usr/bin/env python3
"""pilot_ex_dump.py — corpus evidence for the Act-1 classification pilot.
Input work/pilot_ex_sel.json: for each layer and example (opera / prime) the mass top-10 SHARED features and the
member domains. For every feature this pulls from the 500-prompt captures of ALL domains:
  facts   per member domain: fire rate over 500 prompts, mean coverage (share of a prompt's tokens it fires on, when it fires),
          share of fires that land on the first non-bos token; corpus-wide (all domains): distinct token strings among its
          fires, the six most frequent token strings with their share of fires, and whether fires continue a word
          (token has no leading space / newline and is alphabetic) -- shares only, the judge sees them as numbers.
  other   one more full passage from a member domain where the feature fires (highest peak, prompt != the example prompt).
  (the selection may hold pseudo-examples lang_pJ / code_pJ = prompt set J of the population sample; same format)
  random  fires from NRAND random prompts of the corpus, grouped into language vs code prompts, each fire with 4 tokens of
          context on either side, so the judge sees where else it fires without seeing whole texts.
  texts   per example and member domain: the example's tokens and the fires of its selected features (the card builder
          marks them; the worked examples no longer depend on tokdump_all.json)
Output work/pilot_ex_dump.json  {L: {"cards": {id: {...}}, "texts": {example: {domain: [tokens, {f: {pos: act}}]}}}}
Run on a compute node (CPU, about 60 GB RAM).
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../lib"))  # thesis/analysis/lib
from paths import WORK, capture
from shards import load_domains
import json, collections, time
import numpy as np
SEL = json.load(open(f"{WORK}/pilot_ex_sel.json"))
NRAND = 24; CTX = 3; TOPTOK = 12
CODE = lambda d: d.startswith(("par_code", "div_code", "code_"))
rng = np.random.default_rng(11)
def iscont(tok):
    return tok[:1].isalpha() and tok.isalpha()          # no leading space, purely alphabetic -> continues a word
out = {}
for L, exs in SEL.items():
    t0 = time.time()
    feats = sorted({e["f"] for ex in exs.values() for e in ex["features"]})
    fs = np.array(feats, dtype=np.int64)
    cap = load_domains(capture("27b"), L)
    # per feature: fires[(domain, j)] = {pos: act}  (pos 0 = first non-bos token)
    fires = collections.defaultdict(dict); ntok = {}
    for d, dd in cap.items():
        for j in range(len(dd["token_strings"])):
            toks = dd["token_strings"][j][1:]; ntok[(d, j)] = len(toks)
            pos, fid, act = (np.asarray(x) for x in dd["activations"][j])
            k = (pos > 0) & (pos < len(toks) + 1) & np.isfinite(act)
            pos, fid, act = pos[k].astype(int), fid[k].astype(np.int64), act[k].astype(float)
            s = np.isin(fid, fs)
            if not s.any(): continue
            per = collections.defaultdict(dict)
            for q, f, a in zip(pos[s], fid[s], act[s]):
                if a > per[f].get(int(q) - 1, 0): per[f][int(q) - 1] = a
            for f, h in per.items(): fires[int(f)][(d, j)] = h
    rec = {}; texts = {}
    for ex, cfg in exs.items():
        doms = cfg["domains"]; pj = cfg["prompt"]; exf = [e["f"] for e in cfg["features"]]
        texts[ex] = {d: [cap[d]["token_strings"][pj][1:], {f: {int(k): round(v, 1) for k, v in fires.get(f, {}).get((d, pj), {}).items()} for f in exf}] for d in doms}
        for e in cfg["features"]:
            f = e["f"]; F = fires.get(f, {})
            facts = {"domains": {}}
            for d in doms:
                hits = [(j, h) for (dd, j), h in F.items() if dd == d]
                facts["domains"][d] = {"rate": len(hits) / 500.0,
                                       "cov": float(np.mean([len(h) / ntok[(d, j)] for j, h in hits])) if hits else 0.0,
                                       "first": float(np.mean([0 in h for j, h in hits])) if hits else 0.0}
            tc = collections.Counter(); ncont = 0; nf = 0
            for (d, j), h in F.items():
                toks = cap[d]["token_strings"][j][1:]
                for q in h:
                    tc[toks[q]] += 1; nf += 1; ncont += iscont(toks[q])
            facts["corpus_prompts"] = len(F); facts["corpus_rate"] = len(F) / (500.0 * len(cap))
            facts["distinct_tokens"] = len(tc); facts["nfires"] = nf
            facts["top_tokens"] = [(t, round(c / max(nf, 1), 3)) for t, c in tc.most_common(TOPTOK)]
            facts["top_share"] = sum(c for _, c in tc.most_common(TOPTOK)) / max(nf, 1)
            facts["word_cont"] = ncont / max(nf, 1)
            # other passage: highest peak in a member domain, not the example prompt
            cands = sorted([((dd, j), max(h.values())) for (dd, j), h in F.items() if dd in doms and j != pj], key=lambda r: -r[1])
            other = None
            if cands:
                (d, j), pk = cands[0]
                other = [d, int(j), cap[d]["token_strings"][j][1:], {int(k): round(v, 1) for k, v in F[(d, j)].items()}]
            # random fires from the corpus, language vs code
            keys = [k for k in F if not (k[0] in doms and k[1] == pj)]
            rand = {"lang": [], "code": []}
            for grp in rand:
                pool = [k for k in keys if CODE(k[0]) == (grp == "code")]
                take = rng.choice(len(pool), min(NRAND, len(pool)), replace=False) if pool else []
                for i in take:
                    d, j = pool[i]; toks = cap[d]["token_strings"][j][1:]; h = F[(d, j)]
                    q = max(h, key=h.get)
                    lo, hi = max(0, q - CTX), min(len(toks), q + CTX + 1)
                    rand[grp].append([d, int(j), int(lo), toks[lo:hi], {int(k) - lo: round(v, 1) for k, v in h.items() if lo <= k < hi}])
            rec[f"{ex}_L{L}_f{f}"] = dict(e, example=ex, layer=int(L), facts=facts, other=other, random=rand)
    out[L] = {"cards": rec, "texts": texts}; del cap
    print(f"L{L}: {len(rec)} features  {time.time()-t0:.0f}s", flush=True)
    json.dump(out, open(f"{WORK}/pilot_ex_dump.json", "w"))
print("PILOT_DUMP_DONE", flush=True)
