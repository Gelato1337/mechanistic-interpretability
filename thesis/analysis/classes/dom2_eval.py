#!/usr/bin/env python3
"""dom2_eval.py — the domain-wide judge sample (27 display domains x 19 layers x 4 rank strata x 2 features = 4,104 cards,
dom_sel.py seed 7; Qwen3.5-122B judge with judge/protocol_qwen_v6.txt -> work/qwen_dom2_labels.jsonl; TF statistics
tf_dom2.json). Three questions:
  1 what Qwen says the population is: class shares by rank stratum and by layer (resolved = class in the five, confidence != low)
  2 out-of-par test of the TF classifier: GBM + prevalence matching trained on the 2,334 Sonnet rows (parallel prompts only),
    scored against Qwen on the 4,104 domain-wide cards, overall / per stratum / per layer / per domain group
  3 joint model: 5-fold CV of the GBM on Sonnet + Qwen rows (weights: Sonnet 1, Qwen 1), and Qwen-only CV, as the bridge's
    training set candidates
Output work/dom2_eval.txt and figures/hinge/v4_dom2_shares.png (Qwen shares by layer and stratum)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../lib"))  # thesis/analysis/lib
from paths import WORK, FIGS
import json, collections, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold; from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, cohen_kappa_score
S = WORK; OUT = f"{FIGS}/hinge"
CLS = ["position", "token", "word", "concept", "context"]; CCOL = {"position": "#8c6d31", "token": "#1f77b4", "word": "#e08a1e", "concept": "#d62728", "context": "#e377c2"}
LAB = json.load(open(f"{S}/v3_labels.json")); TF = json.load(open(f"{S}/tf_labelled.json")); TD = json.load(open(f"{S}/tf_dom2.json")); IDX = json.load(open(f"{S}/dom_cards2/index.json"))
Q = {}
for line in open(f"{S}/qwen_dom2_labels.jsonl"):
    r = json.loads(line); Q[r["id"]] = r
STATS = [s for s in next(iter(TF.values())) if s not in ("topstr", "topword")] + ["layer"]
def feats(T, k, L): return [T[k][s] if T[k][s] is not None else np.nan for s in STATS[:-1]] + [L]
def grp(d):
    if d.startswith("par_lang") or d.startswith("disjoint_lang"): return "lang"
    if d.startswith("par_code") or d.startswith("div_code") or d.startswith("code_"): return "code"
    return "content"
STR = ["top100", "top1000", "top5000", "tail"]
ids_s = [k for k, v in LAB.items() if v["resolved"] and v["class"] in CLS and k in TF and TF[k]["nf"] is not None]
Xs = np.nan_to_num(np.array([feats(TF, k, LAB[k]["layer"]) for k in ids_s], float)); ys = np.array([CLS.index(LAB[k]["class"]) for k in ids_s])
def qcls(r): c = r.get("class"); return CLS.index(c) if c in CLS and r.get("confidence") != "low" else -1
ids_q = [k for k in IDX if k in Q and k in TD and TD[k]["nf"] is not None]
yq_all = np.array([qcls(Q[k]) for k in ids_q]); res = yq_all >= 0
Xq = np.nan_to_num(np.array([feats(TD, k, IDX[k]["layer"]) for k in ids_q], float)); yq = yq_all[res]; Xqr = Xq[res]; idr = [k for k, r in zip(ids_q, res) if r]
rep = [f"dom2 cards {len(IDX)} judged {len(Q)} with stats {len(ids_q)} resolved {res.sum()} ({res.mean():.0%}); unparsed {sum(1 for r in Q.values() if r.get('class') is None)}; Qwen raw classes {dict(collections.Counter(r.get('class') for r in Q.values()))}"]
# 1 shares
def share(mask): b = np.bincount(yq_all[mask & res], minlength=5); return b / max(1, b.sum()), (mask & ~res).sum() / max(1, mask.sum())
strat = np.array([IDX[k]["stratum"] for k in ids_q]); lay = np.array([IDX[k]["layer"] for k in ids_q]); gr = np.array([grp(IDX[k]["domain"]) for k in ids_q])
rep.append("\n== 1 Qwen class shares (resolved cards) | unresolved share")
for st in STR:
    s, u = share(strat == st); rep.append(f"  {st:8s} n {int((strat == st).sum()):4d}  " + " ".join(f"{c[:3]} {v:.2f}" for c, v in zip(CLS, s)) + f" | unres {u:.2f}")
for g in ["lang", "code", "content"]:
    s, u = share(gr == g); rep.append(f"  {g:8s} n {int((gr == g).sum()):4d}  " + " ".join(f"{c[:3]} {v:.2f}" for c, v in zip(CLS, s)) + f" | unres {u:.2f}")
LAYS = sorted(set(lay)); SH = np.array([share(lay == L)[0] for L in LAYS]); UN = [share(lay == L)[1] for L in LAYS]
for L, s, u in zip(LAYS, SH, UN): rep.append(f"  L{L:2d} n {int((lay == L).sum()):4d}  " + " ".join(f"{c[:3]} {v:.2f}" for c, v in zip(CLS, s)) + f" | unres {u:.2f}")
fig, axes = plt.subplots(1, 5, figsize=(17, 3.2), sharey=True)
for j, st in enumerate(STR + ["all"]):
    ax = axes[j]; m = (strat == st) if st != "all" else np.ones(len(strat), bool); base = np.zeros(len(LAYS))
    Sh = np.array([share((lay == L) & m)[0] for L in LAYS])
    for c, cl in enumerate(CLS): ax.fill_between(LAYS, base, base + Sh[:, c], color=CCOL[cl], lw=0, label=cl if j == 0 else None); base += Sh[:, c]
    ax.set_ylim(0, 1); ax.set_xlim(0, 61); ax.axvspan(22, 29, color="#f5deb3", alpha=.35, lw=0, zorder=0); ax.set_title(f"{st} (n={int(m.sum())})", fontsize=9); ax.set_xlabel("layer")
axes[0].legend(fontsize=7, frameon=False, loc="lower left"); axes[0].set_ylabel("share of resolved cards")
fig.suptitle("Qwen judge (v6 protocol) class shares of the domain-wide sample by rank stratum: 27 display domains x 19 layers x 2 features per stratum, unresolved cards excluded", x=.01, ha="left", fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/v4_dom2_shares.png", dpi=130); plt.close(fig)
# 2 out-of-par
def gbm(): return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, random_state=0)
def fit_prev(X, y):
    m = gbm().fit(X, y); inner = np.zeros((len(y), 5))
    for itr, ite in StratifiedKFold(4, shuffle=True, random_state=1).split(X, y): inner[ite] = gbm().fit(X[itr], y[itr]).predict_proba(X[ite])
    w = np.ones(5); target = np.bincount(y, minlength=5) / len(y)
    for _ in range(50): got = np.bincount((inner * w).argmax(1), minlength=5) / len(y); w *= (target / np.maximum(got, 1e-3)) ** 0.5
    return m, w
def report(tag, yt, yp):
    Pr, R, F, _ = precision_recall_fscore_support(yt, yp, labels=range(5), zero_division=0); M = confusion_matrix(yt, yp, labels=range(5))
    st = np.bincount(yt, minlength=5) / len(yt); sp = np.bincount(yp, minlength=5) / len(yt)
    rep.append(f"  {tag}: n {len(yt)} acc {np.mean(yt == yp):.3f} kappa {cohen_kappa_score(yt, yp):.3f} mean|share err| {np.abs(st - sp).mean():.3f}  " + " ".join(f"{CLS[c][:3]} P{Pr[c]:.2f}/R{R[c]:.2f}" for c in range(5)))
    return M
m, w = fit_prev(Xs, ys); Pq = m.predict_proba(Xqr) * w; pq = Pq.argmax(1); pq_raw = m.predict(Xqr)
rep.append("\n== 2 out-of-par: GBM trained on Sonnet rows (parallel domains), scored against Qwen on the domain-wide cards")
M = report("raw GBM", yq, pq_raw); M = report("prevalence-matched", yq, pq)
rep.append("   confusion rows=Qwen cols=GBM(prev): " + str(M.tolist()))
sq = np.array([IDX[k]["stratum"] for k in idr]); lq = np.array([IDX[k]["layer"] for k in idr]); gq = np.array([grp(IDX[k]["domain"]) for k in idr])
for st in STR: report(f"  {st:8s}", yq[sq == st], pq[sq == st])
for g in ["lang", "code", "content"]: report(f"  {g:8s}", yq[gq == g], pq[gq == g])
for L in LAYS: report(f"  L{L:2d}     ", yq[lq == L], pq[lq == L])
# 3 joint
rep.append("\n== 3 CV of training-set candidates (5-fold, prevalence-matched GBM)")
def cvrun(tag, X, y, src):
    p = np.zeros(len(y), int)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        mm, ww = fit_prev(X[tr], y[tr]); p[te] = (mm.predict_proba(X[te]) * ww).argmax(1)
    report(tag + " all", y, p)
    for s in sorted(set(src)): report(tag + f" on {s}", y[src == s], p[src == s])
cvrun("Qwen-only", Xqr, yq, np.array(["qwen"] * len(yq)))
cvrun("Sonnet+Qwen", np.r_[Xs, Xqr], np.r_[ys, yq], np.array(["sonnet"] * len(ys) + ["qwen"] * len(yq)))
open(f"{S}/dom2_eval.txt", "w").write("\n".join(rep)); print("\n".join(rep))
