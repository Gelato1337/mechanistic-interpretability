#!/usr/bin/env python3
"""topic_passages3.py SIZE — passages for the topic-feature superset lists (work/topic_super_{size}_L{L}.json) from the
captures of scripts/capture.py, in the record format of the earlier dataset: per feature, counts
per domain, the fired-token count of every prompt it fires in, up to 3 home passages and 2 passages per other domain (most fired tokens
first), each as the token list with the activation on every fired token (start token already excluded in the capture, cut at 320 tokens).
Output work/topic_passages3_{size}_L{L}.json."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../lib"))  # thesis/analysis/lib
from paths import WORK, capture_dirs
import sys, json, glob, time, collections, numpy as np
from transformers import AutoTokenizer
SIZE = sys.argv[1]
LAYERS = {"270m": [9, 10, 11, 12, 13, 14, 15, 16], "1b": [13, 14, 16, 18, 19, 21, 23, 24], "4b": [17, 19, 21, 23, 25, 28, 30, 32], "12b": [24, 27, 30, 33, 36, 39, 42, 45], "27b": [31, 35, 39, 43, 47, 51, 55, 59]}
S = WORK; MAXTOK, PER_HOME, PER_SPILL = 320, 3, 2
DROP = {"div_lang_english", "div_lang_finnish", "div_lang_french", "div_lang_german"}
SRC = [(R, np.load(f"{R}/tokens.npz")) for R in capture_dirs(SIZE)]; R = SRC[0][0]   # one or more capture directories, read as one prompt list
ROWBASE = [int(x) for x in np.cumsum([0] + [int(t["n_tokens"].sum()) for _, t in SRC])[:-1]]
ntok = np.concatenate([t["n_tokens"].astype(np.int64) for _, t in SRC]); dom = np.concatenate([t["domains"] for _, t in SRC]); pid = np.concatenate([t["prompt_ids"] for _, t in SRC]); tokids = np.concatenate([t["token_ids"] for _, t in SRC])
NP = len(ntok); off = np.concatenate([[0], np.cumsum(ntok)]); keep_p = np.array([d not in DROP for d in dom])
tok = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt"); conv = lambda ids: tok.convert_ids_to_tokens(ids.tolist())
def strings(p): return [t.replace("▁", " ") for t in conv(tokids[off[p]:off[p + 1]])][:MAXTOK]
for L in LAYERS[SIZE]:
    t0 = time.time(); sel = json.load(open(f"{S}/topic_super_{SIZE}_L{L}.json"))["features"]; want = np.array(sorted(r["f"] for r in sel)); home = {r["f"]: r["home"] for r in sel}
    if len(want) == 0: json.dump({"size": SIZE, "layer": L, "features": []}, open(f"{S}/topic_passages3_{SIZE}_L{L}.json", "w")); continue
    hits = {int(f): collections.defaultdict(list) for f in want}; wset = np.zeros(262144, bool); wset[want] = True
    for (Rx, _), base in zip(SRC, ROWBASE):
        for fn in sorted(glob.glob(f"{Rx}/layer_{L}/part_*.npz")):
            z = np.load(fn); fe = z["feats"]; m = wset[fe]
            if not m.any(): continue
            rows = z["rows"][m].astype(np.int64) + base; fe = fe[m]; va = z["vals"][m]; p = np.searchsorted(off, rows, side="right") - 1; km = keep_p[p]; rows, fe, va, p = rows[km], fe[km], va[km], p[km]
            order = np.lexsort((rows, p, fe)); fe, p, rows, va = fe[order], p[order], rows[order], va[order]
            b = np.flatnonzero(np.diff(np.concatenate([[-1], fe * NP + p]))); b = np.concatenate([b, [len(fe)]])
            for i in range(len(b) - 1):
                s, e = b[i], b[i + 1]; f = int(fe[s]); pp = int(p[s]); pos = rows[s:e] - off[pp]; hits[f][dom[pp]].append((int(e - s), float(va[s:e].max()), pp, {int(q): float(a) for q, a in zip(pos, va[s:e])}))
    def passage(d, nfire, peak, pp, fires): return dict(prompt=int(pid[pp]), peak=peak, n_tokens=int(ntok[pp]), n_fired=nfire, tokens=strings(pp), fires={str(q): a for q, a in fires.items() if q < MAXTOK})
    out = []
    for f in want.tolist():
        h = hits[f]; hd = home[f]; rec = dict(f=f, home=hd, counts={d: len(v) for d, v in h.items()}, n_fired={d: [x[0] for x in v] for d, v in h.items()}, passages={})
        rec["passages"][hd] = [passage(hd, *x) for x in sorted(h[hd], key=lambda x: -x[0])[:PER_HOME]]
        for d, v in h.items():
            if d != hd and len(v) >= 2: rec["passages"][d] = [passage(d, *x) for x in sorted(v, key=lambda x: -x[0])[:PER_SPILL]]
        out.append(rec)
    json.dump({"size": SIZE, "layer": L, "features": out}, open(f"{S}/topic_passages3_{SIZE}_L{L}.json", "w"), ensure_ascii=False); print(SIZE, L, len(out), round(time.time() - t0), "s", flush=True)
print("PASSAGES3_DONE", SIZE)
