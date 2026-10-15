#!/usr/bin/env python3
"""heur_shared.py — every shared feature of every parallel set, all 62 layers, bucketed by corpus-fact heuristics (no filter).
Sets: lang = par_lang_{english,finnish,french,german,swedish} prompts 0-49 (FLORES), code = par_code_{python,javascript,cpp,c}
prompts 0-49, read from the 27B capture (27B resid_post). Shared = fires on >= 1 token (bos excluded) of every text of
the set. Score of a shared feature in a set = mean over texts of (sum of activations over tokens 2..n / tokens) -- mass with the
first real token excluded (it is the second sink); cov = mean share of tokens fired on.
Corpus facts per feature, over ALL fires in all 500 x 27 prompts of the layer (bos excluded):
  rate      share of the 13,500 prompts in which it fires
  first     share of its fires on the first real token
  bound     share of its fires on a structural slot: first token, or the previous token ends a sentence / line (. ! ? ; : " \\n)
            and the fired token starts a new word (leading space) or a new line -- so `.length` in code is not a slot
  cont      share of its fires on a token that continues a word (no leading space, alphabetic)   [fact only, not a class]
  ndist     distinct token strings among its fires;  top12 = share of fires covered by its 12 most frequent strings
  loud      per (set, feature): the token string on which the feature is loudest in each text (stripped, lower-cased)
Buckets per (set, feature), in this order (first match wins):
  machinery  rate >= MACH            fires in most prompts of every kind; classified separately, excluded from shared counts
  rare       nfires < RARE           too few corpus fires to bucket or judge
  position   first >= FIRST          the first-token sink
  token      top12 >= TOK and the loud string identical in every text of the set, or all loud strings in one family
             (punctuation / whitespace / byte tokens, or digits)
  position   bound >= BOUND          a structural slot decides (first token of a sentence / line), the string does not
  wordcand   top12 >= TOK but the loud strings differ across the texts (energy / Energie / energi): judged as undetermined
  undetermined  the rest (concept or context; the judge decides at snapshot layers)
Output work/heur_shared.json  {"sets": {group: {L: [[f, score, cov, bucket, [loud per text]] per prompt, score desc]}},
                                         "facts": {L: {f: {rate, first, bound, cont, ndist, top12, top: [[tok, share] x12]}}},
                                         "thresholds": {...}}
Run on a compute node (CPU).
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../lib"))  # thesis/analysis/lib
from paths import WORK, capture
from shards import load_domains, layers
import glob, re, json, collections, time
import numpy as np
G = {"lang": ["par_lang_english", "par_lang_finnish", "par_lang_french", "par_lang_german", "par_lang_swedish"],
     "code": ["par_code_python", "par_code_javascript", "par_code_cpp", "par_code_c"]}
NPR = 50; TOPK = 12
TH = {"MACH": 0.6, "RARE": 30, "FIRST": 0.9, "BOUND": 0.8, "TOK": 0.5}
BOUND_END = tuple('.!?;:"') + ("\n",)
vocab = {}; vflag = []                     # token string -> id; per id: (is_boundary_end, is_cont, has_newline, starts_word)
def tid(t):
    i = vocab.get(t)
    if i is None:
        i = len(vocab); vocab[t] = i
        s = t.rstrip(" ")
        vflag.append((s.endswith(BOUND_END) or "\n" in t, t[:1].isalpha() and t.isalpha(), "\n" in t, t[:1] in (" ", "\n", "\t")))
    return i
def bucket(fa, loud):
    if fa["rate"] >= TH["MACH"]: return "machinery"
    if fa["nfires"] < TH["RARE"]: return "rare"
    if fa["first"] >= TH["FIRST"]: return "position"
    fam = lambda l: "punct" if not any(ch.isalnum() for ch in l) else ("digit" if l.isdigit() else l)   # '' / whitespace / bytes -> punct
    same = len(set(fam(l) for l in loud)) == 1
    if fa["top12"] >= TH["TOK"] and same: return "token"
    if fa["bound"] >= TH["BOUND"]: return "position"
    if fa["top12"] >= TH["TOK"]: return "wordcand"
    return "undetermined"
out = {"sets": {g: {} for g in G}, "facts": {}, "thresholds": TH}
for L in layers(capture("27b")):
    t0 = time.time()
    cap = load_domains(capture("27b"), L)
    # 1. shared features per set, with per-text mass (tokens 2..n) and coverage
    sets = {g: [] for g in G}; want = set()
    for g, dl in G.items():
        for j in range(NPR):
            per = {}
            for d in dl:
                toks = cap[d]["token_strings"][j][1:]; n = len(toks)
                pos, fid, act = (np.asarray(x) for x in cap[d]["activations"][j])
                k = (pos > 0) & (pos <= n) & np.isfinite(act); pos, fid, act = pos[k].astype(int), fid[k].astype(np.int64), act[k].astype(float)
                fs = np.unique(fid)
                order = np.lexsort((-act, fid)); lf, li = fid[order], np.searchsorted(fid[order], fs)   # loudest fire per feature
                loud = dict(zip(fs.tolist(), (toks[q - 1].strip().lower() for q in pos[order][li])))
                m = np.bincount(fid[pos > 1], weights=act[pos > 1], minlength=fs.max() + 1)[fs] / n if n > 1 else np.zeros(len(fs))
                key = fid * (n + 2) + pos; c = np.unique(key); cnt = np.bincount(c // (n + 2), minlength=fs.max() + 1)[fs] / n
                per[d] = {f: (mm, cc, loud[f]) for f, mm, cc in zip(fs.tolist(), m.tolist(), cnt.tolist())}
            sh = set.intersection(*[set(v) for v in per.values()]); want |= sh
            rows = [[f, float(np.mean([per[d][f][0] for d in dl])), float(np.mean([per[d][f][1] for d in dl])), None, [per[d][f][2] for d in dl]] for f in sh]
            rows.sort(key=lambda r: -r[1]); sets[g].append(rows)
    W = np.zeros(262144, bool); W[list(want)] = True
    # 2. corpus facts for every wanted feature over all fires of the layer
    F, T, PV, FIRSTF = [], [], [], []; npr_fire = collections.Counter(); ntot = 0
    for d, dd in cap.items():
        for j in range(len(dd["token_strings"])):
            ntot += 1; toks = dd["token_strings"][j]; n = len(toks) - 1
            ids = np.fromiter((tid(t) for t in toks), dtype=np.int64, count=len(toks))
            pos, fid, act = (np.asarray(x) for x in dd["activations"][j])
            k = (pos > 0) & (pos <= n) & np.isfinite(act) & W[fid.astype(np.int64)]
            if not k.any(): continue
            pos, fid = pos[k].astype(int), fid[k].astype(np.int64)
            u = np.unique(fid * (n + 2) + pos); fid, pos = u // (n + 2), u % (n + 2)   # one fire per (feature, position)
            F.append(fid); T.append(ids[pos]); PV.append(ids[pos - 1]); FIRSTF.append(pos == 1)
            npr_fire.update(np.unique(fid).tolist())
    F, T, PV, FIRSTF = (np.concatenate(x) for x in (F, T, PV, FIRSTF))
    vb, vc, vn, vs = (np.array([v[i] for v in vflag], bool) for i in range(4))
    bound = FIRSTF | (vb[PV] & (vs[T] | vn[PV])); cont = vc[T]
    nf = np.bincount(F, minlength=262144); first_n = np.bincount(F[FIRSTF], minlength=262144)
    bound_n = np.bincount(F[bound], minlength=262144); cont_n = np.bincount(F[cont], minlength=262144)
    V = len(vocab); pair, pc = np.unique(F * V + T, return_counts=True); pf = pair // V; pt = pair % V
    order = np.lexsort((-pc, pf)); pf, pt, pc = pf[order], pt[order], pc[order]
    starts = np.searchsorted(pf, np.array(sorted(want)))
    inv = {i: t for t, i in vocab.items()}
    facts = {}
    for f, s in zip(sorted(want), starts):
        e = np.searchsorted(pf, f, side="right"); c = pc[s:e]; n_ = nf[f]
        if n_ == 0: continue
        fa = {"rate": npr_fire[f] / ntot, "first": first_n[f] / n_, "bound": bound_n[f] / n_, "cont": cont_n[f] / n_,
              "ndist": int(e - s), "top12": float(c[:TOPK].sum() / n_), "nfires": int(n_),
              "top": [[inv[int(t)], round(float(k) / n_, 3)] for t, k in zip(pt[s:s + TOPK], c[:TOPK])]}
        facts[int(f)] = fa
    for g in G:
        for rows in sets[g]:
            for r in rows: r[3] = bucket(facts[r[0]], r[4]) if r[0] in facts else "rare"
    out["sets"]["lang"][L] = sets["lang"]; out["sets"]["code"][L] = sets["code"]; out["facts"][L] = facts
    cnt = {g: collections.Counter(r[3] for rows in sets[g] for r in rows) for g in G}
    print(f"L{L:2d} {time.time()-t0:5.0f}s fires {len(F)/1e6:5.1f}M feats {len(facts):6d} | lang {dict(cnt['lang'])} | code {dict(cnt['code'])}", flush=True)
    del cap
    json.dump(out, open(f"{WORK}/heur_shared.json", "w"))
print("HEUR_DONE", flush=True)
