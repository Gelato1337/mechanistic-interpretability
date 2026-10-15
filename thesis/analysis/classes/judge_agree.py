#!/usr/bin/env python3
"""judge_agree.py — agreement of the local Qwen3.5-122B judge (work/qwen_labels.jsonl, judge_vllm.py) with the Sonnet
labels (work/v3_labels.json) on the same 3,325 cards. Sonnet = reference. Classes position/token/word/concept/context
+ unresolved (other / low confidence). Reports accuracy, Cohen's kappa, per-class precision/recall (Qwen vs Sonnet, resolved
cards of both), the confusion matrix, and the same split by track (shared / machinery / pop). Figure figures/hinge/
v3_qwen_vs_sonnet.png. Also writes work/qwen_agree.json."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../lib"))  # thesis/analysis/lib
from paths import WORK, FIGS
import json, os, collections, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = WORK; OUT = f"{FIGS}/hinge"
CLS = ["position", "token", "word", "concept", "context", "unres"]
SON = json.load(open(f"{S}/v3_labels.json")); Q = {}
QF = os.environ.get("QWEN", f"{S}/qwen_labels.jsonl"); TAG = os.environ.get("TAG", "")   # env QWEN: other label file, TAG: output suffix
for line in open(QF):
    r = json.loads(line); Q[r["id"]] = r
def qcls(r):
    c = r.get("class"); return c if c in CLS[:-1] and r.get("confidence") != "low" else "unres"
def scls(v): return v["class"] if v["resolved"] and v["class"] in CLS[:-1] else "unres"   # 'global' (machinery track) counted as unresolved here
ids = [k for k in SON if k in Q]; print("cards", len(SON), "qwen labelled", len(Q), "both", len(ids), "unparsed", sum(Q[k].get("class") is None for k in ids))
def report(ids, name):
    M = np.zeros((6, 6), int)
    for k in ids: M[CLS.index(scls(SON[k])), CLS.index(qcls(Q[k]))] += 1
    n = M.sum(); acc = np.trace(M) / n; pe = (M.sum(1) * M.sum(0)).sum() / n / n; kap = (acc - pe) / (1 - pe)
    both = [k for k in ids if scls(SON[k]) != "unres" and qcls(Q[k]) != "unres"]
    M5 = np.zeros((5, 5), int)
    for k in both: M5[CLS.index(scls(SON[k])), CLS.index(qcls(Q[k]))] += 1
    acc5 = np.trace(M5) / max(1, M5.sum()); pe5 = (M5.sum(1) * M5.sum(0)).sum() / max(1, M5.sum()) ** 2; kap5 = (acc5 - pe5) / (1 - pe5)
    pr = {c: {"P": M5[i, i] / max(1, M5[:, i].sum()), "R": M5[i, i] / max(1, M5[i].sum()), "n_sonnet": int(M5[i].sum()), "n_qwen": int(M5[:, i].sum())} for i, c in enumerate(CLS[:-1])}
    print(f"\n== {name}: n={n}  acc(6 cls)={acc:.3f} kappa={kap:.3f} | both resolved n={len(both)} ({len(both)/n:.0%}) acc={acc5:.3f} kappa={kap5:.3f}")
    print("  Sonnet unresolved", int(M[5].sum()), " Qwen unresolved", int(M[:, 5].sum()))
    print("  class     P     R   n_sonnet n_qwen"); [print(f"  {c:8s} {v['P']:.2f}  {v['R']:.2f}  {v['n_sonnet']:5d} {v['n_qwen']:5d}") for c, v in pr.items()]
    print("  confusion rows=Sonnet cols=Qwen:", CLS); [print("  ", f"{CLS[i]:8s}", M[i]) for i in range(6)]
    return {"n": int(n), "acc6": acc, "kappa6": kap, "n_both": len(both), "acc5": acc5, "kappa5": kap5, "pr": pr, "M": M.tolist()}
res = {"all": report(ids, "ALL")}
for tr in ("shared", "machinery", "pop"): res[tr] = report([k for k in ids if SON[k]["track"] == tr], tr)
json.dump(res, open(f"{S}/qwen_agree{TAG}.json", "w"), indent=1, default=float)
fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))
for ax, (name, r) in zip(axes, res.items()):
    M = np.array(r["M"]); Mn = M / np.maximum(1, M.sum(1, keepdims=True)); ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1)
    for i in range(6):
        for j in range(6): ax.text(j, i, str(M[i, j]), ha="center", va="center", fontsize=8, color="white" if Mn[i, j] > .5 else "black")
    ax.set_xticks(range(6)); ax.set_xticklabels(CLS, rotation=45, ha="right", fontsize=8); ax.set_yticks(range(6)); ax.set_yticklabels(CLS, fontsize=8)
    ax.set_xlabel("Qwen3.5-122B (local)"); ax.set_ylabel("Sonnet (reference)")
    ax.set_title(f"{name}: n={r['n']}  acc={r['acc6']:.2f} κ={r['kappa6']:.2f}\nboth resolved: n={r['n_both']} acc={r['acc5']:.2f} κ={r['kappa5']:.2f}", fontsize=9)
fig.suptitle("Judge agreement on the same cards (rows Sonnet, cols Qwen, row-normalised colour; unres = other or low confidence)", fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/v3_qwen_vs_sonnet{TAG}.png", dpi=130); print("fig ok")
