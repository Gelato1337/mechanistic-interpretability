#!/usr/bin/env python3
"""dom_dump.py (CPU, about 60 GB RAM) — corpus evidence for the domain-wide judge sample (work/dom_sel.json,
dom_sel.py): the same evidence as pilot_ex_dump_v3.py, but the "example" of a feature is its HOME domain: the NEX passages of
that domain where it peaks highest are the example texts, the next one is the other passage; random fires from the rest of the
corpus (language vs code prompts) as before. facts: home-domain rate / coverage / first-token share, corpus rate, distinct
strings, top strings, word-continuation share.
Output work/dom_dump.json {L: {id: {f, domain, rank, n, stratum, mass, facts, texts: [[domain, j, tokens, {pos: act}]], other, random}}}
id = dom_<domain>_L<L>_f<f>."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../lib"))  # thesis/analysis/lib
from paths import WORK, capture, UNUSED_DIVERSE, SWEDISH_DIVERSE
from shards import load_domains
import os, json, collections, time
import numpy as np
SEL = json.load(open((os.environ.get("SEL") or f"{WORK}/dom_sel.json"))); OUTF = (os.environ.get("OUT") or f"{WORK}/dom_dump.json")
DROP = UNUSED_DIVERSE | {SWEDISH_DIVERSE}   # corpus = the 27 display domains
NRAND = 12; CTX = 3; TOPTOK = 12; NEX = 2
CODE = lambda d: d.startswith(("par_code", "div_code", "code_"))
rng = np.random.default_rng(11)
def iscont(tok): return tok[:1].isalpha() and tok.isalpha()
out = {}
for L, byd in SEL.items():
    t0 = time.time(); feats = sorted({r["f"] for rows in byd.values() for r in rows}); fs = np.array(feats, dtype=np.int64)
    cap = load_domains(capture("27b"), L); cap = {d: v for d, v in cap.items() if d not in DROP}
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
    rec = {}
    for d, rows in byd.items():
        for e in rows:
            f = e["f"]; F = fires.get(f, {})
            hits = [(j, h) for (dd, j), h in F.items() if dd == d]
            facts = {"domains": {d: {"rate": len(hits) / 500.0, "cov": float(np.mean([len(h) / ntok[(d, j)] for j, h in hits])) if hits else 0.0,
                                     "first": float(np.mean([0 in h for j, h in hits])) if hits else 0.0}}}
            tc = collections.Counter(); ncont = 0; nf = 0
            for (dd, j), h in F.items():
                toks = cap[dd]["token_strings"][j][1:]
                for q in h: tc[toks[q]] += 1; nf += 1; ncont += iscont(toks[q])
            facts["corpus_prompts"] = len(F); facts["corpus_rate"] = len(F) / (500.0 * len(cap)); facts["distinct_tokens"] = len(tc); facts["nfires"] = nf
            facts["top_tokens"] = [(t, round(c / max(nf, 1), 3)) for t, c in tc.most_common(TOPTOK)]; facts["top_share"] = sum(c for _, c in tc.most_common(TOPTOK)) / max(nf, 1)
            facts["word_cont"] = ncont / max(nf, 1)
            home = sorted(hits, key=lambda r: -max(r[1].values()))
            texts = [[d, int(j), cap[d]["token_strings"][j][1:], {int(k): round(v, 1) for k, v in h.items()}] for j, h in home[:NEX]]
            other = [d, int(home[NEX][0]), cap[d]["token_strings"][home[NEX][0]][1:], {int(k): round(v, 1) for k, v in home[NEX][1].items()}] if len(home) > NEX else None
            used = {(d, j) for j, _ in home[:NEX + 1]}; keys = [k for k in F if k not in used]
            rand = {"lang": [], "code": []}
            for grp in rand:
                pool = [k for k in keys if CODE(k[0]) == (grp == "code")]
                take = rng.choice(len(pool), min(NRAND, len(pool)), replace=False) if pool else []
                for i in take:
                    dd, j = pool[i]; toks = cap[dd]["token_strings"][j][1:]; h = F[(dd, j)]; q = max(h, key=h.get)
                    lo, hi = max(0, q - CTX), min(len(toks), q + CTX + 1)
                    rand[grp].append([dd, int(j), int(lo), toks[lo:hi], {int(k) - lo: round(v, 1) for k, v in h.items() if lo <= k < hi}])
            rec[f"dom_{d}_L{L}_f{f}"] = dict(e, domain=d, layer=int(L), facts=facts, texts=texts, other=other, random=rand)
    out[L] = rec; del cap
    print(f"L{L}: {len(rec)} cards  {time.time()-t0:.0f}s", flush=True)
    json.dump(out, open(OUTF, "w"))
print("DOM_DUMP_DONE", flush=True)
