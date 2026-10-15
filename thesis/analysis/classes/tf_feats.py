#!/usr/bin/env python3
"""tf_feats.py — term-frequency model of every feature: per (layer, feature) statistics of the distribution of the token strings
it fires on, against the corpus background, for the heuristic-vs-judge study (target: P/R >= .8 per class vs judge labels).
usage: tf_feats.py <L0> <L1>   (layers L0..L1-1; run_tf_feats.sh = array of 8)
Data: the 27B capture (scripts/capture.py; paths.capture('27b')), read through shards.load_domains; <bos> excluded; one fire per
(feature, position).
Token strings are used raw (vocab id per distinct string); q(t) = corpus frequency of string t over all 16,500 prompts.
Per feature f (all 262,144, arrays of length D; NaN where the feature never fires):
  nf, rate (share of prompts fired), fpp (fires per fired prompt), dens (mean over fired prompts of fires / tokens),
  first, bound, cont (as bridge_v3), top1, top12 (share of fires on the 1 / 12 most frequent strings), ndist,
  h_tok  entropy (bits) of p_f(t);  kl_tok = KL(p_f || q) (bits);  idf = sum_t p_f(t) log2(N / count(t))
  h_prev, kl_prev: the same for the PREVIOUS token's string (position features: previous-token distribution concentrated)
  rec_top  = fires on the top string / corpus occurrences of that string  (token / word features: high; context: low)
  rec_mean = sum_t p_f(t) * count_f(t) / count(t)   (weighted corpus recall over all its strings)
  loc_rec  = sum over fired prompts of fires on string t / occurrences of t in that prompt, over the strings fired (local recall)
  run      = mean run length of consecutive fired positions (context features fire in runs)
  relpos   = mean position / prompt length;  early3 = share of fires on positions 1..3
  type shares of fires: ty_ws (space + alphabetic = word start), ty_cont (alphabetic continuation), ty_punct, ty_digit, ty_nl
  (contains newline), ty_other; kl_type = KL of that 6-way mix from the corpus mix
  stem = share of fires whose stripped-lowercased string shares its first 4 letters with the top string (inflections)
  h_dom = entropy (bits) over the domains of the prompts fired; dom_top = share of fired prompts in the top domain; code_share
  act_mean, act_cv (std / mean of the activations of its fires); lmax = max activation
  WORD units (pieces merged into the containing word: a token continues the word when it is alphabetic without leading
  whitespace and the previous token ends in a letter; word string = stripped, lowercased): nw = word occurrences with >= 1 fire,
  pieces = nf / nw, ndist_w, top1_w, top12_w, h_word, kl_word (vs corpus word-occurrence frequency), idf_w, rec_wtop (word
  occurrences fired of the top word / all its occurrences), rec_wmean; topword = the top word string
  NEIGHBOURHOOD (the NB=3 tokens either side of a fire, fired position excluded): h_nb, kl_nb vs the corpus token background —
  concept / token features have a weak neighbourhood signal, context features a strong one
  Per domain (domains x D): MASS = mean over the domain's prompts of (sum of activations over tokens 2..n / tokens) (bridge_v3 rank),
  NFD = prompts fired
Output work/tf_feats/L{L}.npz (all features) — the labelled rows are cut out by tf_extract.py."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../lib"))  # thesis/analysis/lib
from paths import WORK, capture
from shards import load_domains
import sys, os, time, collections
import numpy as np
D = 262144; TOPK = 12; NB = 3
BOUND_END = tuple('.!?;:"') + ("\n",)
OUT = f"{WORK}/tf_feats"; os.makedirs(OUT, exist_ok=True)
L0, L1 = int(sys.argv[1]), int(sys.argv[2])
def ent(pf, p):                                  # entropy per feature from (feature id, probability) pairs
    return np.bincount(pf, weights=-p * np.log2(np.maximum(p, 1e-300)), minlength=D)
for L in range(L0, L1):
    if os.path.exists(f"{OUT}/L{L}.npz"): print("skip", L, flush=True); continue
    t0 = time.time(); cap = load_domains(capture("27b"), L)
    names = sorted(cap); n = len(names); code_dom = np.array(["code" in nm for nm in names])
    vocab = {}; vflag = []; wvocab = {}; wcount = collections.Counter()
    def wid(w):
        i = wvocab.get(w)
        if i is None: i = len(wvocab); wvocab[w] = i
        return i
    def tid(t):
        i = vocab.get(t)
        if i is None:
            i = len(vocab); vocab[t] = i; s = t.rstrip(" "); tn = t.strip().lower(); a = t[:1]
            ty = 4 if "\n" in t else (0 if (a in (" ", "\t") and t[1:2].isalpha()) else (1 if a.isalpha() else (2 if tn and not any(ch.isalnum() for ch in tn) else (3 if tn.isdigit() else 5))))
            stem = "".join(ch for ch in tn if ch.isalpha())[:4]
            vflag.append((s.endswith(BOUND_END) or "\n" in t, "\n" in t, a in (" ", "\n", "\t"), ty, stem))
        return i
    NF = np.zeros((n, D), np.int32); DENS = np.zeros(D); ADJ = np.zeros(D, np.int64); LOCF = np.zeros(D, np.int64); LOCO = np.zeros(D, np.int64)
    RELP = np.zeros(D); EARLY = np.zeros(D, np.int64); ASUM = np.zeros(D); ASQ = np.zeros(D); LMAX = np.zeros(D, np.float32); MASS = np.zeros((n, D)); NWF = np.zeros(D, np.int64)
    F, T, PV, FIRSTF, FW, FWF, FNB = [], [], [], [], [], [], []; npr_fire = np.zeros(D, np.int64); ntot = 0; qcount = collections.Counter(); bcount = collections.Counter()
    NPR = []
    for i, nm in enumerate(names):
        dd = cap[nm]; npr = len(dd["token_strings"]); NPR.append(npr)
        for j in range(npr):
            toks = dd["token_strings"][j]; nt = len(toks) - 1; ntot += 1
            ids = np.fromiter((tid(t) for t in toks), dtype=np.int64, count=len(toks)); qcount.update(ids[1:].tolist()); bcount[ids[0]] += 1
            wocc = np.zeros(len(toks), np.int64); wstr = []; cur = -1                     # word occurrence index and word id per position
            for p in range(1, len(toks)):
                t = toks[p]; cont_ = p > 1 and t[:1].isalpha() and toks[p - 1][-1:].isalpha()
                if cont_: wstr[-1] += t.strip().lower()
                else: cur += 1; wstr.append(t.strip().lower())
                wocc[p] = cur
            wids = np.array([wid(w) for w in wstr], np.int64); wcount.update(wids.tolist()); widp = np.r_[-1, wids[wocc[1:]]]
            padded = np.r_[np.full(NB, -1), ids[1:], np.full(NB, -1)]                     # neighbourhood lookup, -1 = outside the prompt
            pos, fid, act = (np.asarray(x) for x in dd["activations"][j])
            k = (pos > 0) & (pos <= nt) & np.isfinite(act)
            pos, fid, act = pos[k].astype(np.int64), fid[k].astype(np.int64), act[k].astype(np.float64)
            if len(fid) == 0: continue
            u, ui = np.unique(fid * (nt + 2) + pos, return_index=True); fid, pos, act = fid[ui], pos[ui], act[ui]   # sorted by (fid, pos)
            cnt = np.bincount(fid, minlength=D); fired = cnt > 0; NF[i, fired] += 1; npr_fire += fired; DENS += cnt / nt
            m2 = pos > 1; MASS[i] += np.bincount(fid[m2], weights=act[m2], minlength=D) / nt
            wo = wocc[pos]; wfirst = np.r_[True, (fid[1:] != fid[:-1]) | (wo[1:] != wo[:-1])]      # first fire of f inside a word occurrence (sorted by fid, pos)
            NWF += np.bincount(fid[wfirst], minlength=D); FW.append(widp[pos].astype(np.int32)); FWF.append(wfirst)
            nb = np.stack([padded[pos - 1 + NB + o] for o in list(range(-NB, 0)) + list(range(1, NB + 1))], 1); FNB.append(nb.astype(np.int32))
            adj = (fid[1:] == fid[:-1]) & (pos[1:] - pos[:-1] == 1); ADJ += np.bincount(fid[1:][adj], minlength=D)
            RELP += np.bincount(fid, weights=pos / nt, minlength=D); EARLY += np.bincount(fid[pos <= 3], minlength=D)
            ASUM += np.bincount(fid, weights=act, minlength=D); ASQ += np.bincount(fid, weights=act * act, minlength=D)
            tt = ids[pos]; VV = len(vocab); key, kc = np.unique(fid * VV + tt, return_counts=True); kf, kt = key // VV, key % VV
            pc = np.bincount(ids[1:]); LOCF += np.bincount(kf, weights=kc, minlength=D).astype(np.int64); LOCO += np.bincount(kf, weights=pc[kt], minlength=D).astype(np.int64)
            uf, st = np.unique(fid, return_index=True); mx = np.maximum.reduceat(act, st).astype(np.float32); LMAX[uf] = np.maximum(LMAX[uf], mx)
            F.append(fid.astype(np.int32)); T.append(tt.astype(np.int32)); PV.append(ids[pos - 1].astype(np.int32)); FIRSTF.append(pos == 1)
    del cap
    F, T, PV, FIRSTF, FW, FWF, FNB = (np.concatenate(x) for x in (F, T, PV, FIRSTF, FW, FWF, FNB)); F = F.astype(np.int64); V = len(vocab)
    for i in range(n): MASS[i] /= NPR[i]
    vb, vn, vs = (np.array([v[i] for v in vflag], bool) for i in (0, 1, 2)); vty = np.array([v[3] for v in vflag], np.int64)
    stemid = {}; vstem = np.array([stemid.setdefault(v[4], len(stemid)) if v[4] else -1 for v in vflag], np.int64)
    Q = np.zeros(V);
    for k, v in qcount.items(): Q[k] = v
    NTOK = Q.sum(); q = Q / NTOK; QB = Q.copy()
    for k, v in bcount.items(): QB[k] += v                       # previous-token background includes <bos>
    nf = np.bincount(F, minlength=D).astype(np.float64); nfz = np.maximum(nf, 1)
    bound = FIRSTF | (vb[PV] & (vs[T] | vn[PV]))
    first = np.bincount(F[FIRSTF], minlength=D) / nfz; bnd = np.bincount(F[bound], minlength=D) / nfz
    tyF = vty[T]; TY = np.stack([np.bincount(F[tyF == c], minlength=D) / nfz for c in range(6)], 1)   # ws, cont, punct, digit, nl, other
    qty = np.array([Q[vty == c].sum() for c in range(6)]) / NTOK
    kl_type = (TY * np.log2(np.maximum(TY, 1e-12) / np.maximum(qty, 1e-12))).sum(1)
    def dist_stats(X, Q):                                         # X: string id per fire (token or previous token); Q its background counts
        NTOK = Q.sum(); q = Q / NTOK; pair, pc = np.unique(F * V + X.astype(np.int64), return_counts=True); pf = pair // V; pt = pair % V
        o = np.lexsort((-pc, pf)); pf, pc, pt = pf[o], pc[o], pt[o]; p = pc / nfz[pf]
        h = ent(pf, p); kl = np.bincount(pf, weights=p * np.log2(p / q[pt]), minlength=D)
        idf = np.bincount(pf, weights=p * np.log2(NTOK / Q[pt]), minlength=D); rec = np.bincount(pf, weights=p * pc / Q[pt], minlength=D)
        ndist = np.bincount(pf, minlength=D); starts = np.searchsorted(pf, np.arange(D)); ends = np.searchsorted(pf, np.arange(D), side="right")
        top = np.zeros(D); topid = np.full(D, -1, np.int64); ok0 = starts < ends; topid[ok0] = pt[starts[ok0]]
        for r in range(TOPK):
            idx = starts + r; ok = idx < ends; top[ok] += pc[idx[ok]]
            if r == 0: top1 = top.copy()
        return pf, pc, pt, p, h, kl, idf, rec, ndist, top1 / nfz, top / nfz, topid, starts, ends
    pf, pc, pt, p, h_tok, kl_tok, idf, rec_mean, ndist, top1, top12, topid, starts, ends = dist_stats(T, Q)
    rec_top = np.zeros(D); ok = topid >= 0; rec_top[ok] = pc[starts[ok]] / Q[topid[ok]]
    same = vstem[pt] >= 0; same &= vstem[pt] == vstem[np.maximum(topid[pf], 0)]; stem = np.bincount(pf[same], weights=pc[same], minlength=D) / nfz
    _, _, _, _, h_prev, kl_prev, _, _, _, _, _, _, _, _ = dist_stats(PV, QB)
    # word units: one count per (feature, word occurrence); background = word occurrence counts
    W = len(wvocab); QW = np.zeros(W)
    for k, v in wcount.items(): QW[k] = v
    Fs, Vs, nfzs = F, V, nfz
    F = Fs[FWF]; V = W; nfz = np.maximum(NWF, 1).astype(np.float64)
    _, _, _, _, h_word, kl_word, idf_w, rec_wmean, ndist_w, top1_w, top12_w, topwid, wstarts, _ = dist_stats(FW[FWF], QW)
    pw, pcw = np.unique(F * V + FW[FWF].astype(np.int64), return_counts=True); pfw = pw // V; ow = np.lexsort((-pcw, pfw)); pfw, pcw = pfw[ow], pcw[ow]
    rec_wtop = np.zeros(D); okw = topwid >= 0; rec_wtop[okw] = pcw[np.searchsorted(pfw, np.arange(D))[okw]] / QW[topwid[okw]]
    F, V, nfz = Fs, Vs, nfzs
    # neighbourhood: the 2*NB tokens around each fire
    nbm = FNB.reshape(-1) >= 0; Fnb = np.repeat(F, 2 * NB)[nbm]; Xnb = FNB.reshape(-1)[nbm].astype(np.int64); nfz_nb = np.maximum(np.bincount(Fnb, minlength=D), 1).astype(np.float64)
    Fs, nfzs = F, nfz; F, nfz = Fnb, nfz_nb
    _, _, _, _, h_nb, kl_nb, _, _, _, _, _, _, _, _ = dist_stats(Xnb, Q)
    F, nfz = Fs, nfzs; del Fnb, Xnb, FNB
    cont = np.bincount(F[(tyF == 1)], minlength=D) / nfz
    rate = npr_fire / ntot; fpp = nf / np.maximum(npr_fire, 1); dens = DENS / np.maximum(npr_fire, 1)
    run = nf / np.maximum(nf - ADJ, 1); loc_rec = LOCF / np.maximum(LOCO, 1); relpos = RELP / nfz; early3 = EARLY / nfz
    act_mean = ASUM / nfz; act_cv = np.sqrt(np.maximum(ASQ / nfz - act_mean ** 2, 0)) / np.maximum(act_mean, 1e-9)
    pd = NF / np.maximum(NF.sum(0), 1); h_dom = (-pd * np.log2(np.maximum(pd, 1e-12))).sum(0); dom_top = pd.max(0); code_share = pd[code_dom].sum(0)
    fired = nf > 0
    out = dict(nf=nf, rate=rate, fpp=fpp, dens=dens, first=first, bound=bnd, cont=cont, top1=top1, top12=top12, ndist=ndist.astype(np.float64),
               h_tok=h_tok, kl_tok=kl_tok, idf=idf, h_prev=h_prev, kl_prev=kl_prev, rec_top=rec_top, rec_mean=rec_mean, loc_rec=loc_rec, run=run,
               relpos=relpos, early3=early3, ty_ws=TY[:, 0], ty_cont=TY[:, 1], ty_punct=TY[:, 2], ty_digit=TY[:, 3], ty_nl=TY[:, 4], ty_other=TY[:, 5],
               kl_type=kl_type, stem=stem, h_dom=h_dom, dom_top=dom_top, code_share=code_share, act_mean=act_mean, act_cv=act_cv, lmax=LMAX.astype(np.float64),
               nw=NWF.astype(np.float64), pieces=nf / np.maximum(NWF, 1), ndist_w=ndist_w.astype(np.float64), top1_w=top1_w, top12_w=top12_w, h_word=h_word, kl_word=kl_word,
               idf_w=idf_w, rec_wtop=rec_wtop, rec_wmean=rec_wmean, h_nb=h_nb, kl_nb=kl_nb)
    for k in out: out[k] = np.where(fired, out[k], np.nan).astype(np.float32)
    inv = [None] * V
    for s, i in vocab.items(): inv[i] = s
    topstr = np.array([inv[i] if i >= 0 else "" for i in topid], dtype=object); winv = [None] * W
    for s_, i in wvocab.items(): winv[i] = s_
    topword = np.array([winv[i] if i >= 0 else "" for i in topwid], dtype=object)
    np.savez_compressed(f"{OUT}/L{L}.npz", topstr=topstr, topword=topword, names=np.array(names), qty=qty, MASS=MASS.astype(np.float32), NFD=NF, **out)
    print(f"L{L}: {int(fired.sum())} features, vocab {V}, fires {len(F)}, {time.time()-t0:.0f}s", flush=True)
print("TF_FEATS_DONE")
