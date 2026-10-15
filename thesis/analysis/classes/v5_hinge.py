#!/usr/bin/env python3
"""v5_hinge.py — the three hinge figures of story v5 (27B, 28 display domains, <bos> excluded, machinery = corpus rate >= .6 removed
and said so; every other feature carries its class from the judge-calibrated classifier, features seen < 30 times folded in).
  h1_impact       firing vs impact: cumulative share of a domain's activation mass against feature rank (work/cummass.json)
  h2_classes      the domain average by class, three ways: mass share / count share of all fired / count share of the top-5,000
                  (dom_profile_v6.json for mass and count; bridge_v4_v6 diagonal for top-5,000); columns English / other languages / code / content
  h3_reliability  judge vs judge (qwen_agree_sub_v6.json), classifier vs judge (5-fold CV recomputed as tf_fluff_v6.py), share error per layer band
Output figures/hinge/v5_h{1,2,3}_*.png; numbers appended to work/v4_story.txt."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../lib"))  # thesis/analysis/lib
from paths import WORK, FIGS
import notitle   # thesis figures carry no header
import json, glob, re, os, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = WORK; OUT = f"{FIGS}/hinge"
CLS = ["position", "token", "word", "concept", "context"]; CCOL = {"position": "#8c6d31", "token": "#1f77b4", "word": "#e08a1e", "concept": "#d62728", "context": "#e377c2"}
LANGS = ["english", "finnish", "french", "german", "swedish"]; CODES = ["python", "javascript", "cpp", "c"]
DIV = {l: (f"div_lang_{l}" if l == "swedish" else f"disjoint_lang_{l}") for l in LANGS}
G = {"English": ["par_lang_english", DIV["english"]],
     "other languages": [f"par_lang_{l}" for l in LANGS[1:]] + [DIV[l] for l in LANGS[1:]],
     "code": [f"par_code_{c}" for c in CODES] + [f"div_code_{c}" for c in CODES] + ["code_sql", "code_assembly"],
     "content": ["fact_common", "fact_science", "reasoning_legal", "reasoning_math", "reasoning_moral", "story_fiction", "abstract_philosophy", "abstract_religion"]}
GCOL = {"English": "#1f77b4", "other languages": "#7fb3d5", "code": "#e08a1e", "content": "#3a9d5d"}; BAND = (22, 28)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 9.5}); rep = []
def band(ax):
    for x in BAND: ax.axvline(x, color="k", lw=.7, ls=":", zorder=5)
def norm(v): v = np.asarray(v, float); return v / max(v.sum(), 1e-12)

# ---------- H1: firing vs impact ----------
C = json.load(open(f"{S}/cummass{os.environ.get('SFX', '_v2')}.json")); LC = sorted(int(l) for l in C)
fig, axes = plt.subplots(1, len(LC), figsize=(3.2 * len(LC), 3.6), sharey=True)
for ax, L in zip(axes, LC):
    E = C[str(L)]; ns, t5, rr, rm = [], [], [], []
    for g, doms in G.items():
        for d in doms:
            r = E[d]; ax.plot(r["ranks"], r["cum"], color=GCOL[g], lw=1, alpha=.75, label=g if d == doms[0] else None, marker="o", ms=2)
            ns.append(r["n"]); t5.append(r["cum"][r["ranks"].index(5000)] if 5000 in r["ranks"] else 1.0); rr.append(r["n_rare"] / r["n"]); rm.append(r["mass_rare"])
    for x in (1000, 5000): ax.axvline(x, color="k", lw=.7, ls=":")
    ax.set_xscale("log"); ax.set_ylim(0, 1.02); ax.set_xlim(1, 6e4); ax.grid(alpha=.25); ax.set_xlabel("features that fire, ranked by mass")
    ax.set_title(f"L{L} · {min(ns):,}–{max(ns):,} features fire", loc="left")
    ax.text(.03, .97, f"top-5,000 carry {min(t5):.2f}–{max(t5):.2f} of the mass\nseen < 30×: {min(rr):.2f}–{max(rr):.2f} of the features,\n{min(rm):.3f}–{max(rm):.3f} of the mass", transform=ax.transAxes, va="top", fontsize=7.5, color="#333", linespacing=1.4)
    rep.append(f"H1impact L{L:2d} fired {min(ns):,}-{max(ns):,}; top-1,000 carries {min(E[d]['cum'][E[d]['ranks'].index(1000)] for d in sum(G.values(), [])):.2f}-{max(E[d]['cum'][E[d]['ranks'].index(1000)] for d in sum(G.values(), [])):.2f}, top-5,000 {min(t5):.2f}-{max(t5):.2f}; seen<30x {min(rr):.2f}-{max(rr):.2f} of features, {min(rm):.3f}-{max(rm):.3f} of mass")
axes[0].set_ylabel("cumulative share of the domain's\nactivation mass"); axes[0].legend(fontsize=7.5, frameon=False, loc="lower right")
fig.suptitle("Firing vs impact · 27B · one line per domain (28 domains × 500 prompts) · a feature fires when its activation is > 0; what it writes into the residual stream has length = activation (decoder rows are unit-norm)\n"
             "mass = that length summed over tokens 2..n, per token, averaged over the domain's prompts · machinery removed · dotted = ranks 1,000 and 5,000", x=.01, ha="left", fontsize=8.5)
fig.tight_layout(rect=(0, 0, 1, .88)); fig.savefig(f"{OUT}/v5_h1_impact.png", dpi=140); plt.close(fig)

# ---------- H2: the domain average by class, three ways ----------
D = json.load(open(f"{S}/dom_profile_v6{os.environ.get('SFX', '_v2')}.json"))["layers"]; LD = sorted(int(l) for l in D)
R = {int(re.search(r"L(\d+)", os.path.basename(p)).group(1)): json.load(open(p)) for p in glob.glob(f"{S}/bridge_v4_v6{os.environ.get('SFX', '_v2')}/L*.json")}; LB = sorted(R)
def mass(L, d): r = D[str(L)][d]; return norm(np.array(r["mass"][2:]) + np.array(r["rare_mass"]))
def count(L, d): r = D[str(L)][d]; return norm(np.array(r["count"][2:]) + np.array(r["rare_count"]))
def rare_count(L, d): r = D[str(L)][d]; return r["count"][1] / max(1 - r["count"][0], 1e-9)
def top5(L, d):
    b = R[L]; i = b["names"].index(d); return norm([b["top5000"][c][i][i] + b["top5000"][f"rare_{c}"][i][i] for c in CLS])
ROWS = [("what carries the residual stream\nshare of the mass, all fired features", mass, LD), ("what fires\nshare of the features, all fired", count, LD), ("the top-5,000 by mass\nshare of the features", top5, LB)]
fig, axes = plt.subplots(3, 4, figsize=(14, 8.2), sharex=True, sharey=True)
for c, (g, doms) in enumerate(G.items()):
    for r, (lab, fn, LAY) in enumerate(ROWS):
        ax = axes[r, c]; X = np.mean([[fn(L, d) for L in LAY] for d in doms], 0); base = np.zeros(len(LAY))
        for j, cl in enumerate(CLS): ax.fill_between(LAY, base, base + X[:, j], color=CCOL[cl], lw=0, label=cl); base += X[:, j]
        if r == 1:
            rl = np.mean([[rare_count(L, d) for L in LAY] for d in doms], 0); ax.plot(LAY, rl, color="k", ls="--", lw=1.2, label="share seen < 30 times in the corpus", zorder=6)
        ax.set_ylim(0, 1); ax.set_xlim(0, 61); ax.grid(alpha=.25); band(ax)
        if r == 0: ax.set_title(f"{g} · mean of {len(doms)} domains", loc="left")
        if r == 2: ax.set_xlabel("layer (27B resid_post)")
        for L in (2, 24, 42, 61): k = LAY.index(L); rep.append(f"H2classes {g:15s} row{r + 1} L{L:2d} " + " ".join(f"{cl[:3]} {X[k, j]:.2f}" for j, cl in enumerate(CLS)) + (f" rare {rl[k]:.2f}" if r == 1 else ""))
for r, (lab, _, _) in enumerate(ROWS): axes[r, 0].set_ylabel(lab)
h, l = axes[1, 0].get_legend_handles_labels(); fig.legend(h, l, ncol=6, fontsize=8.5, frameon=False, loc="upper left", bbox_to_anchor=(.04, .955))
fig.suptitle("The domain average by class, counted three ways · 27B · 28 domains × 500 prompts · machinery (corpus rate ≥ .6) removed · every other feature carries its class from the classifier · dotted = layers 22–28", x=.01, ha="left", fontsize=9.5)
fig.tight_layout(rect=(0, 0, 1, .92)); fig.savefig(f"{OUT}/v5_h2_classes.png", dpi=140); plt.close(fig)

# ---------- H3: reliability ----------
from sklearn.ensemble import HistGradientBoostingClassifier; from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, cohen_kappa_score
Q = json.load(open(f"{S}/qwen_agree_sub_v6.json"))["all"]; MQ = np.array(Q["M"])[:5, :5]
TR = json.load(open(f"{S}/tf_train.json")); STATS = TR["stats"]; rows = TR["rows"]
X = np.nan_to_num(np.array([[r["x"].get(s, np.nan) if r["x"].get(s) is not None else np.nan for s in STATS] for r in rows], float)); y = np.array([CLS.index(r["y"]) for r in rows]); lay = X[:, STATS.index("layer")]
gbm = lambda: HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, random_state=0)
def fit_prev(X, y):
    m = gbm().fit(X, y); inner = np.zeros((len(y), 5))
    for itr, ite in StratifiedKFold(4, shuffle=True, random_state=1).split(X, y): inner[ite] = gbm().fit(X[itr], y[itr]).predict_proba(X[ite])
    w = np.ones(5); target = np.bincount(y, minlength=5) / len(y)
    for _ in range(50): got = np.bincount((inner * w).argmax(1), minlength=5) / len(y); w *= (target / np.maximum(got, 1e-3)) ** 0.5
    return m, w
p = np.zeros(len(y), int)
for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
    m, w = fit_prev(X[tr], y[tr]); p[te] = (m.predict_proba(X[te]) * w).argmax(1)
MC = confusion_matrix(y, p, labels=range(5)); accC, kC = float(np.mean(p == y)), cohen_kappa_score(y, p)
def cm(ax, M, title, xl, yl):
    F = M / M.sum(1, keepdims=True); ax.imshow(F, cmap="Blues", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5): ax.text(j, i, f"{F[i, j]:.2f}\n({M[i, j]})", ha="center", va="center", fontsize=7.5, color="white" if F[i, j] > .5 else "#1a1a1a")
    ax.set_xticks(range(5)); ax.set_yticks(range(5)); ax.set_xticklabels(CLS, rotation=25, ha="right", fontsize=8.5); ax.set_yticklabels(CLS); ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title, loc="left", fontsize=9)
    for s in ax.spines.values(): s.set_visible(True)
fig, axes = plt.subplots(1, 3, figsize=(14, 5.3), gridspec_kw={"width_ratios": [1, 1, 1.5]})
cm(axes[0], MQ, f"judge vs judge\n{MQ.sum()} cards both resolved · agreement {Q['acc5']:.2f} · κ {Q['kappa5']:.2f}", "Qwen 3.5 122B says", "Sonnet says")
cm(axes[1], MC, f"classifier vs judge\n{len(y):,} judged features, 5-fold CV · accuracy {accC:.2f} · κ {kC:.2f}", "classifier says", "judge says")
BANDS = [("L0–7", 0, 7), ("L8–21", 8, 21), ("L22–28", 22, 28), ("L29–45", 29, 45), ("L46–61", 46, 61)]; ax = axes[2]; wdt = .15; errs = []
for bi, (bl, lo, hi) in enumerate(BANDS):
    m = (lay >= lo) & (lay <= hi); sj = np.bincount(y[m], minlength=5) / m.sum(); sp = np.bincount(p[m], minlength=5) / m.sum(); e = sp - sj; errs.append(np.abs(e).max())
    for j, cl in enumerate(CLS): ax.bar(bi + (j - 2) * wdt, e[j], wdt, color=CCOL[cl], label=cl if bi == 0 else None)
    ax.text(bi, .085, f"n {m.sum():,}", ha="center", fontsize=7, color="#555")
ax.axhline(0, color="k", lw=.6); ax.set_xticks(range(len(BANDS))); ax.set_xticklabels([b[0] for b in BANDS]); ax.set_ylim(-.1, .1); ax.grid(alpha=.25, axis="y")
ax.set_ylabel("predicted share − judged share"); ax.set_title(f"the number the figures inherit\nclass share error per layer band · worst {max(errs):.3f}", loc="left", fontsize=9); ax.legend(ncol=5, fontsize=7, frameon=False, loc="lower left")
fig.suptitle("How reliable the five classes are · left: two judges on the same feature cards (Sonnet reference) · middle and right: the classifier that labels every feature,\nout of fold against the judge labels (2,334 Sonnet + 3,067 Qwen rows) · rows sum to 1, counts in brackets", x=.01, ha="left", fontsize=9)
fig.tight_layout(rect=(0, .02, 1, .9)); fig.savefig(f"{OUT}/v5_h3_reliability.png", dpi=140); plt.close(fig)
# ---------- the same, two panels (Methods version) ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1, 1.4]})
cm(axes[0], MC, "", "classifier", "judge")
BANDS = [("L0–7", 0, 7), ("L8–21", 8, 21), ("L22–28", 22, 28), ("L29–45", 29, 45), ("L46–61", 46, 61)]; ax = axes[1]; wdt = .15; errs = []
for bi, (bl, lo, hi) in enumerate(BANDS):
    m = (lay >= lo) & (lay <= hi); sj = np.bincount(y[m], minlength=5) / m.sum(); sp = np.bincount(p[m], minlength=5) / m.sum(); e = sp - sj; errs.append(np.abs(e).max())
    for j, cl in enumerate(CLS): ax.bar(bi + (j - 2) * wdt, e[j], wdt, color=CCOL[cl], label=cl if bi == 0 else None)
ax.axhline(0, color="k", lw=.6); ax.set_xticks(range(len(BANDS))); ax.set_xticklabels([b[0] for b in BANDS]); ax.set_ylim(-.1, .1); ax.grid(alpha=.25, axis="y")
ax.set_ylabel("predicted share − judged share"); ax.legend(ncol=5, fontsize=8, frameon=False, loc="lower left")
fig.tight_layout(rect=(0, .04, 1, 1)); fig.savefig(f"{OUT}/v5_h3_reliability2.png", dpi=140); plt.close(fig)
P = np.diag(MC) / MC.sum(0); Rc = np.diag(MC) / MC.sum(1)
rep.append(f"H3 judge-vs-judge n {MQ.sum()} acc {Q['acc5']:.3f} kappa {Q['kappa5']:.3f}; classifier CV acc {accC:.3f} kappa {kC:.3f}; P/R " + " ".join(f"{c[:3]} {P[j]:.2f}/{Rc[j]:.2f}" for j, c in enumerate(CLS)) + f"; worst band share error {max(errs):.3f}; string-vs-meaning agreement {(MC[:3, :3].sum() + MC[3:, 3:].sum()) / MC.sum():.3f}")
open(f"{S}/v4_story.txt", "a").write("\n" + "\n".join(rep) + "\n"); print("\n".join(r for r in rep if not r.startswith("H2classes") or " L24 " in r or " L61 " in r)); print("hinge ok")
