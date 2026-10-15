#!/usr/bin/env python3
"""v4_story.py — the simple figures of story v4: the two hinge figures (activation mass by class; top-K sets by class) and the
plain Act 2 / Act 3 curves at the same three K's (top-1,000, top-5,000, all fired). 27 display domains, <bos> excluded,
nothing filtered; 28 display domains (the diverse-language tier of paths.DIVERSE_LANG); shared machinery (corpus rate >= .6) removed from the Jaccard and drawn as its own band; every other feature
carries one of the five classes of the judge-calibrated TF classifier (bridge_v4_v6) — features seen < 30 times in the corpus
("rare" in bridge_v4) are folded into their class and their share is drawn as a dashed line, not as a band.
Inputs  work/dom_profile_v6.json (dom_profile.py), work/bridge_v4_v6/L*.json (bridge_v4.py),
        work/ent_v4_{size}.json (ent_v4.py: Act 3 under the same rules, five sizes)
Output  figures/hinge/h1_mass.png, hinge/h2_topk.png, act2/a2_mcurve.png, act2/a2_mat.png, act2/a2_mat_big.png, act2/a2_cells.png,
        act3/a3_mcurve.png, act3/a3_ladder.png, act3/a3_cells.png, act3/a3_gain.png, act3/a3_twoway.png
        work/v4_story.txt (the numbers quoted in the page)"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../lib"))  # thesis/analysis/lib
from paths import WORK, FIGS, DIVERSE_LANG, UNUSED_DIVERSE
import json, glob, re, os, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
# 2026-10-10 thesis figures: NOTITLE=1 (default) drops every suptitle and every long axes title (panel labels such as "L24" stay); everything goes in the caption.
if os.environ.get("NOTITLE", "1") == "1":
    import matplotlib.figure, matplotlib.axes
    matplotlib.figure.Figure.suptitle = lambda self, *a, **k: None
    _st = matplotlib.axes.Axes.set_title
    matplotlib.axes.Axes.set_title = lambda self, label="", *a, **k: (_st(self, label, *a, **k) if len(str(label)) <= 32 else None)
S = WORK; TAG = os.environ.get("TAG", "_v6"); V2 = os.environ.get("V2", "1") == "1"
B = f"{S}/bridge_v4{TAG}" + ("_v2" if V2 else ""); OUT = FIGS
SFX = "_v2" if V2 else ""   # 2026-09-16 v2 data: JumpReLU fixed (activations = pre-activation), 14 philosophy/religion duplicates dropped, split halves
CLS = ["position", "token", "word", "concept", "context"]; CCOL = {"position": "#8c6d31", "token": "#1f77b4", "word": "#e08a1e", "concept": "#d62728", "context": "#e377c2"}
BCOL = {"machinery": "#444444", "rare": "#b8e0b8", **CCOL}
LANGS = ["english", "finnish", "french", "german", "swedish"]; CODES = ["python", "javascript", "cpp", "c"]; BAND = (22, 29)
# the diverse-language tier (paths.DIVERSE_LANG)
DIV = dict(zip(LANGS, DIVERSE_LANG)); DIVL = [DIV[l] for l in LANGS]
DROP = UNUSED_DIVERSE
KEYS = ["top1000", "top5000", "unbounded"]; KL = {"top1000": "top-1,000", "top5000": "top-5,000", "unbounded": "all fired"}
KST = {"top1000": ("#3a9d5d", "-"), "top5000": ("#1d5c8a", "-"), "unbounded": ("k", "--")}
PAIRCOL = {"same content": "#d62728", "same language": "#1f77b4", "neither": "#7f7f7f"}
GROUPS = {"language": [f"par_lang_{l}" for l in LANGS] + DIVL,
          "code": [f"par_code_{c}" for c in CODES] + [f"div_code_{c}" for c in CODES] + ["code_sql", "code_assembly"],
          "content": ["fact_common", "fact_science", "reasoning_legal", "reasoning_math", "reasoning_moral", "story_fiction", "abstract_philosophy", "abstract_religion"]}
PAR = [(f"par_lang_{a}", f"par_lang_{b}") for i, a in enumerate(LANGS) for b in LANGS[i + 1:]] + [(f"par_code_{a}", f"par_code_{b}") for i, a in enumerate(CODES) for b in CODES[i + 1:]]
CELLS = {"sentences": {"same content": [(f"par_lang_{a}", f"par_lang_{b}") for i, a in enumerate(LANGS) for b in LANGS[i + 1:]],
                       "same language": [(f"par_lang_{l}", DIV[l]) for l in LANGS],
                       "neither": [(f"par_lang_{a}", DIV[b]) for a in LANGS for b in LANGS if a != b]},
         "programs": {"same content": [(f"par_code_{a}", f"par_code_{b}") for i, a in enumerate(CODES) for b in CODES[i + 1:]],
                      "same language": [(f"par_code_{c}", f"div_code_{c}") for c in CODES],
                      "neither": [(f"par_code_{a}", f"div_code_{b}") for a in CODES for b in CODES if a != b]}}
NDOM = sum(len(v) for v in GROUPS.values()); NPAIR = NDOM * (NDOM - 1) // 2   # 28 display domains, 378 pairs
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 9.5})
rep = []
def band(ax, b=BAND): ax.grid(alpha=.25)
def stack(ax, X, LAYERS, cols):
    base = np.zeros(len(LAYERS))
    for j, c in enumerate(cols): ax.fill_between(LAYERS, base, base + X[:, j], color=BCOL[c], lw=0, label=c); base += X[:, j]
    ax.set_ylim(0, 1); ax.set_xlim(0, 61); ax.grid(alpha=.25)
    for x in BAND: ax.axvline(x, color="k", lw=.7, ls=":", zorder=5)
def legend_top(fig, ax, ncol, y=.9):
    h, l = ax.get_legend_handles_labels(); fig.legend(h, l, ncol=ncol, fontsize=8.5, frameon=False, loc="upper left", bbox_to_anchor=(0.05, y))
def norm(C): return C / np.maximum(C.sum(1, keepdims=True), 1e-9)

# ---------- H1: what an average prompt's activation is made of ----------
D = json.load(open(f"{S}/dom_profile{TAG}{SFX}.json")); bands = D["bands"]; LD = sorted(int(l) for l in D["layers"])
SIX = ["machinery"] + CLS
def prof(d):   # mass shares over six bands: machinery, then each class with its < 30-fire features folded in (rare_mass)
    return np.array([[D["layers"][str(L)][d]["mass"][0]] + [D["layers"][str(L)][d]["mass"][2 + j] + D["layers"][str(L)][d]["rare_mass"][j] for j in range(5)] for L in LD])
fig, axes = plt.subplots(2, 3, figsize=(13, 5.8), sharex=True, sharey=True)
for c, (g, doms) in enumerate(GROUPS.items()):
    M = np.mean([prof(d) for d in doms], 0); five = norm(M[:, 1:])
    stack(axes[0, c], M, LD, SIX); axes[0, c].set_title(f"{g} · mean of {len(doms)} domains", loc="left")
    stack(axes[1, c], five, LD, CLS); axes[1, c].set_xlabel("layer (27B resid_post)")
    for L in (2, 7, 24, 42, 61): k = LD.index(L); rep.append(f"H1 {g:9s} L{L:2d} mass " + " ".join(f"{b[:4]} {M[k, j]:.2f}" for j, b in enumerate(SIX)) + " | five " + " ".join(f"{b[:3]} {five[k, j]:.2f}" for j, b in enumerate(CLS)))
axes[0, 0].set_ylabel("share of an average prompt's activation\nevery firing feature"); axes[1, 0].set_ylabel("the five classes only")
legend_top(fig, axes[0, 0], 6, .935)
fig.suptitle(f"Where an average prompt's activation goes, by class · {NDOM} domains × 500 prompts · activation mass over tokens 2..n · dotted = layers 22–29", x=.01, ha="left", fontsize=9.5)
fig.tight_layout(rect=(0, 0, 1, .9)); fig.savefig(f"{OUT}/hinge/h1_mass.png", dpi=140); plt.close(fig)

# ---------- bridge matrices ----------
R = {int(re.search(r"L(\d+)", os.path.basename(p)).group(1)): json.load(open(p)) for p in glob.glob(f"{B}/L*.json")}
LAY = sorted(R); ALL = R[LAY[0]]["names"]; KEEP = [i for i, d in enumerate(ALL) if d not in DROP]; names = [ALL[i] for i in KEEP]; n = len(names); assert n == NDOM, (n, NDOM); ix = {d: i for i, d in enumerate(names)}
iu = np.triu_indices(n, 1); di = (np.arange(n), np.arange(n))
def mat(L, key, what): return np.array(R[L][key][what], float)[np.ix_(KEEP, KEEP)]
JM = {}
def Jmat(L, key):
    if (L, key) not in JM:
        sh = mat(L, key, "shared") - mat(L, key, "machinery"); un = mat(L, key, "union") - mat(L, key, "mach_union"); J = sh / np.maximum(un, 1); np.fill_diagonal(J, np.nan); JM[(L, key)] = J
    return JM[(L, key)]

# ---------- H2: what the top-K sets are, and what two domains share ----------
fig, axes = plt.subplots(3, 2, figsize=(10.5, 7.4), sharex=True, sharey=True)
def counts(key, sel):   # (layers, 5 classes, domains|pairs): every non-machinery feature of the class, < 30-fire features included
    return np.array([[mat(L, key, c)[sel] + mat(L, key, f"rare_{c}")[sel] for c in CLS] for L in LAY])
def rare_share(key, sel): return np.array([(mat(L, key, "rare")[sel] / np.maximum(mat(L, key, "shared")[sel] - mat(L, key, "machinery")[sel], 1)).mean() for L in LAY])
for r, key in enumerate(KEYS):
    Cset, Csh = counts(key, di), counts(key, iu)
    Sset = np.array([norm(C.T).mean(0) for C in Cset]); Ssh = np.array([norm(C.T).mean(0) for C in Csh])   # mean of per-domain / per-pair shares
    Rset, Rsh = rare_share(key, di), rare_share(key, iu); nset, nsh = Cset.sum(1).mean(1), Csh.sum(1).mean(1)
    stack(axes[r, 0], Sset, LAY, CLS); stack(axes[r, 1], Ssh, LAY, CLS)
    for ax, Rr in ((axes[r, 0], Rset), (axes[r, 1], Rsh)): ax.plot(LAY, Rr, color="k", ls="--", lw=1.3, label="share seen < 30 times in the corpus", zorder=6)
    axes[r, 0].set_ylabel(f"{KL[key]}\nshare of the set"); axes[r, 0].set_title(f"a domain's own set · {int(nset.min()):,}–{int(nset.max()):,} features (mean of {NDOM} domains)", loc="left")
    axes[r, 1].set_title(f"shared by two domains · {int(nsh.min()):,}–{int(nsh.max()):,} features (mean of {NPAIR} pairs)", loc="left")
    for L in (2, 7, 24, 42, 61): k = LAY.index(L); rep.append(f"H2 {KL[key]:10s} L{L:2d} set n {int(nset[k]):6d} rare {Rset[k]:.2f} " + " ".join(f"{c[:3]} {Sset[k, j]:.2f}" for j, c in enumerate(CLS)) + f" | shared n {int(nsh[k]):6d} rare {Rsh[k]:.2f} " + " ".join(f"{c[:3]} {Ssh[k, j]:.2f}" for j, c in enumerate(CLS)))
for ax in axes[-1]: ax.set_xlabel("layer (27B resid_post)")
legend_top(fig, axes[0, 0], 6, .95)
fig.suptitle("Each domain's top-K features by mass, and what two domains share, by class (share of the features, by count) · machinery removed · dotted = layers 22–29", x=.01, ha="left", fontsize=9.5)
fig.tight_layout(rect=(0, 0, 1, .92)); fig.savefig(f"{OUT}/hinge/h2_topk.png", dpi=140); plt.close(fig)
rc = np.array([R[L]["n_class_rare"] for L in LAY]); nc = np.array([R[L]["n_class"] for L in LAY]); rep.append("H2 population: fired features per layer " + " ".join(f"L{L} {nc[k].sum():,} ({rc[k].sum() / nc[k].sum():.2f} seen < 30 times; of those token {rc[k, 1] / rc[k].sum():.2f} word {rc[k, 2] / rc[k].sum():.2f} concept {rc[k, 3] / rc[k].sum():.2f})" for k, L in enumerate(LAY) if L in (2, 24, 42, 61)))

# ---------- A2: the M-curve ----------
fig, ax = plt.subplots(figsize=(8, 4))
for key in KEYS:
    Jall = [np.nanmean(Jmat(L, key)[iu]) for L in LAY]; Jpar = [np.mean([Jmat(L, key)[ix[a], ix[b]] for a, b in PAR]) for L in LAY]; col, ls = KST[key]
    ax.plot(LAY, Jall, color=col, ls=ls, lw=2, label=f"{KL[key]} · all {NPAIR} pairs"); ax.plot(LAY, Jpar, color=col, ls=ls, lw=1, alpha=.6, label=f"{KL[key]} · 16 parallel pairs")
    for L in (2, 7, 24, 42, 61): k = LAY.index(L); rep.append(f"A2 {KL[key]:10s} L{L:2d} J all {Jall[k]:.3f} par {Jpar[k]:.3f}")
band(ax); ax.set_ylim(0, .9); ax.set_xlim(0, 61); ax.set_xlabel("layer (27B resid_post)"); ax.set_ylabel("Jaccard of the two domains' feature sets"); ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper left")
ax.set_title(f"Entanglement of the {NDOM} domains by depth · thick = mean over all pairs, thin = the 16 same-content pairs", loc="left")
fig.tight_layout(); fig.savefig(f"{OUT}/act2/a2_mcurve.png", dpi=140); plt.close(fig)

# ---------- A2: the matrices (format of story_figs/act2/fig_entmat3.png) ----------
ORDER = [f"par_lang_{l}" for l in LANGS] + DIVL + [f"par_code_{c}" for c in CODES] + [f"div_code_{c}" for c in CODES] + ["code_sql", "code_assembly"] + \
        ["reasoning_math", "reasoning_moral", "reasoning_legal", "abstract_philosophy", "abstract_religion", "fact_science", "fact_common", "story_fiction"]
BOUNDS = [5, 10, 14, 18, 20]; oi = [ix[d] for d in ORDER]; assert len(ORDER) == n
short = lambda d: d.replace("par_lang_", "par ").replace("disjoint_lang_", "div ").replace("div_lang_", "div ").replace("par_code_", "par ").replace("div_code_", "div ").replace("_", " ")
LAYM = [0, 12, 18, 24, 32, 43, 61]; BLOCKS = "blocks: par lang | diverse lang | par code | diverse code | sql, assembly | content"
fig, axes = plt.subplots(3, len(LAYM), figsize=(2.6 * len(LAYM), 8.4))
for c, L in enumerate(LAYM):
    for r, key in enumerate(KEYS):
        M = Jmat(L, key)[np.ix_(oi, oi)]; ax = axes[r, c]; im = ax.imshow(M, cmap="Blues", vmin=0, vmax=0.8)
        for b in BOUNDS: ax.axhline(b - .5, color="#b45309", lw=.8); ax.axvline(b - .5, color="#b45309", lw=.8)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0: ax.set_title(f"L{L}", fontsize=11)
        if c == 0: ax.set_yticks(range(n)); ax.set_yticklabels([short(d) for d in ORDER], fontsize=4.5); ax.set_ylabel(KL[key] + " by activation mass", fontsize=8)
        if r == 2: ax.set_xticks(range(n)); ax.set_xticklabels([short(d) for d in ORDER], rotation=90, fontsize=4.5)
fig.colorbar(im, ax=axes, shrink=.6, pad=.01, label="Jaccard of the two domains' feature sets")
fig.suptitle(f"27B · 262k · {NDOM} domains × 500 prompts · <bos> excluded · shared machinery removed · top-K by activation mass · Jaccard. {BLOCKS}.", fontsize=8, x=.01, ha="left")
fig.savefig(f"{OUT}/act2/a2_mat.png", dpi=150, bbox_inches="tight"); plt.close(fig)
LAYP = [0, 6, 12, 18, 24, 32, 43, 61]   # one page per K: 2 x 4 layers, big enough to read the domain labels
for key in KEYS:
    fig, axes = plt.subplots(4, 2, figsize=(10, 17), constrained_layout=True)
    for k_, L in enumerate(LAYP):
        ax = axes[k_ // 2, k_ % 2]; M = Jmat(L, key)[np.ix_(oi, oi)]; im = ax.imshow(M, cmap="Blues", vmin=0, vmax=0.8)
        for b in BOUNDS: ax.axhline(b - .5, color="#b45309", lw=.9); ax.axvline(b - .5, color="#b45309", lw=.9)
        ax.set_title(f"L{L}", fontsize=13); ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels([short(d) for d in ORDER] if k_ >= 6 else [], rotation=90, fontsize=6.5); ax.set_yticklabels([short(d) for d in ORDER] if k_ % 2 == 0 else [], fontsize=6.5)
    fig.colorbar(im, ax=axes, shrink=.25, pad=.005, label="Jaccard of the two domains' feature sets")
    fig.savefig(f"{OUT}/act2/a2_mat_{key}.png", dpi=150, bbox_inches="tight"); plt.close(fig)
LB, KB = 24, "top5000"; M = Jmat(LB, KB)[np.ix_(oi, oi)]
fig, ax = plt.subplots(figsize=(14, 13)); im = ax.imshow(M, cmap="Blues", vmin=0, vmax=0.8)
for b in BOUNDS: ax.axhline(b - .5, color="#b45309", lw=1.2); ax.axvline(b - .5, color="#b45309", lw=1.2)
for i in range(n):
    for j in range(n):
        if i != j: ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="white" if M[i, j] > .45 else "#1a1a1a")
ax.set_xticks(range(n)); ax.set_yticks(range(n)); ax.set_xticklabels([short(d) for d in ORDER], rotation=90, fontsize=9); ax.set_yticklabels([short(d) for d in ORDER], fontsize=9)
fig.colorbar(im, ax=ax, shrink=.7, pad=.02, label=f"Jaccard of {KL[KB]} feature sets")
ax.set_title(f"27B · L{LB} · {KL[KB]} by activation mass · {NDOM} domains × 500 prompts · <bos> excluded · shared machinery removed · {BLOCKS}", loc="left", fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/act2/a2_mat_big.png", dpi=150, bbox_inches="tight"); plt.close(fig)
for key in KEYS:
    for L in (2, 24, 42, 61):
        M = Jmat(L, key)[np.ix_(oi, oi)]; blk = [0] + BOUNDS + [n]; within = np.nanmean(np.concatenate([M[a:b, a:b][~np.eye(b - a, dtype=bool)] for a, b in zip(blk[:-1], blk[1:]) if b - a > 1])); between = np.nanmean([M[i, j] for i in range(n) for j in range(n) if i != j and np.searchsorted(BOUNDS, i, "right") != np.searchsorted(BOUNDS, j, "right")])
        rep.append(f"A2mat {KL[key]:10s} L{L:2d} within-block mean J {within:.3f} between-block {between:.3f}")

# ---------- A2: same content / same language / neither ----------
def cells_fig(Jfun, LAYERS, path, keys, title, bands_by_col=None):
    fig, axes = plt.subplots(2, len(keys), figsize=(4.3 * len(keys) + 0.5, 6), sharex="col", sharey=True, squeeze=False)
    for r, (kind, sets) in enumerate(CELLS.items()):
        for c, key in enumerate(keys):
            ax = axes[r, c]
            for lab, prs in sets.items():
                V = np.array([[Jfun(key, L, a, b) for a, b in prs] for L in LAYERS]); ax.plot(LAYERS, V.mean(1), color=PAIRCOL[lab], lw=2, label=f"{lab} ({len(prs)} pairs)"); ax.fill_between(LAYERS, V.min(1), V.max(1), color=PAIRCOL[lab], alpha=.13, lw=0)
                if path.endswith("a2_cells.png"):
                    for L in (2, 24, 42, 61): rep.append(f"A2cells {kind:9s} {KL[key]:10s} {lab:13s} L{L:2d} {V[LAYERS.index(L)].mean():.3f}")
            band(ax, bands_by_col[c] if bands_by_col else BAND); ax.set_ylim(0, .9); ax.set_xlim(0, LAYERS[-1]); ax.set_title(f"{kind} · {key if bands_by_col else KL[key]}", loc="left")
            if r == 1: ax.set_xlabel("layer")
        axes[r, 0].set_ylabel("Jaccard"); axes[r, 0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle(title, x=.01, ha="left", fontsize=9.5); fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(path, dpi=140); plt.close(fig)
cells_fig(lambda key, L, a, b: Jmat(L, key)[ix[a], ix[b]], LAY, f"{OUT}/act2/a2_cells.png", KEYS,
          "Same content (the same 500 texts in two languages) vs same language (other texts, same language) vs neither · mean and min–max over pairs · shared machinery removed")

# ---------- A3: model size, under the Act 2 rules (ent_v4.py: per-token capture, <bos> excluded, machinery removed) ----------
SIZES = ["270m", "1b", "4b", "12b", "27b"]; SL = {"270m": "270M", "1b": "1B", "4b": "4B", "12b": "12B", "27b": "27B"}
E3P = (lambda s_: f"{S}/ent_v4_262k_{s_}.json") if V2 else (lambda s_: f"{S}/ent_v4_{s_}.json")
E3 = {s_: json.load(open(E3P(s_))) for s_ in SIZES if os.path.exists(E3P(s_))}; SIZES = [s_ for s_ in SIZES if s_ in E3]
def JM(E, L, key, thr="0.6", half=None):   # one Jaccard matrix from an ent_v4 file, either layout
    r = E["layers"][str(L)]
    if half is not None: return np.array(r["J_half"][key][half], float)
    return np.array(r["J"][thr][key] if isinstance(r["J"].get("0.6"), dict) else r["J"][key], float)
KL3 = ["top100", "top1000", "top2000", "top5000", "top10000", "top20000", "unbounded"]; KLL = {**KL, "top100": "top-100", "top2000": "top-2,000", "top10000": "top-10,000", "top20000": "top-20,000"}
KCOL = dict(zip(KL3, ["#9ccb3b", "#3a9d5d", "#1f8a80", "#1d5c8a", "#3b3f8f", "#5a2d7a", "k"]))
class Ent3:
    def __init__(self, s_):
        E = E3[s_]; nm = E["names"]; self.keep = [i for i, d in enumerate(nm) if d not in DROP and not d.startswith("lenm_")]; self.ix = {nm[i]: k for k, i in enumerate(self.keep)}
        self.L = sorted(int(l) for l in E["layers"]); self.nl = E["n_layers"]; self.E = E; self.cache = {}
    def J(self, L, key):
        if (L, key) not in self.cache:
            M = JM(self.E, L, key)[np.ix_(self.keep, self.keep)]; np.fill_diagonal(M, np.nan); self.cache[(L, key)] = M
        return self.cache[(L, key)]
    def allpairs(self, key): return np.array([np.nanmean(self.J(L, key)[np.triu_indices(len(self.keep), 1)]) for L in self.L])
    def pairs(self, key, prs): return np.array([[self.J(L, key)[self.ix[a], self.ix[b]] for a, b in prs] for L in self.L])   # (layers, pairs)
ENT = {s_: Ent3(s_) for s_ in SIZES}
def band3(e):   # the band of a size = the longest run of layers where top-5,000 sharing over all pairs is >= 80 % of its peak
    v = e.allpairs("top5000"); ok = v >= .8 * v.max(); best = (0, 0); i = 0
    while i < len(ok):
        if ok[i]:
            j = i
            while j + 1 < len(ok) and ok[j + 1]: j += 1
            if j - i > best[1] - best[0]: best = (i, j)
            i = j + 1
        else: i += 1
    return None if best[1] - best[0] + 1 > len(e.L) / 2 else (e.L[best[0]], e.L[best[1]])   # a run over half the depth is a plateau, not a band
B3 = {s_: band3(ENT[s_]) for s_ in SIZES}
BC = {s_: (ENT[s_].L.index((B3[s_][0] + B3[s_][1]) // 2) if B3[s_] else int(np.argmax(ENT[s_].allpairs("top5000")))) for s_ in SIZES}   # band centre, or the top-5,000 peak layer
BT = {s_: (f"band L{B3[s_][0]}–{B3[s_][1]}" if B3[s_] else "no band") for s_ in SIZES}
def shade3(ax, s_):
    ax.grid(alpha=.25)
# a3_mcurve: the Act 2 M-curve per size, the same three K's
fig, axes = plt.subplots(2, 3, figsize=(11, 7.2), sharey=True); axes = axes.ravel(); axes[-1].axis("off"); axes = axes[:len(SIZES)]
for ax, s_ in zip(axes, SIZES):
    e = ENT[s_]
    for key in KEYS:
        Jall = e.allpairs(key); Jpar = e.pairs(key, PAR).mean(1); col, ls = KST[key]
        ax.plot(e.L, Jall, color=col, ls=ls, lw=2, label=f"{KL[key]} · all {NPAIR} pairs"); ax.plot(e.L, Jpar, color=col, ls=ls, lw=1, alpha=.6, label=f"{KL[key]} · 16 parallel pairs")
        k = int(np.argmax(Jall)); rep.append(f"A3 {SL[s_]:4s} {KL[key]:10s} max J all {Jall[k]:.3f} at L{e.L[k]} of {e.L[-1]} (par {Jpar[k]:.3f}); at band centre / peak {Jall[BC[s_]]:.3f}")
    shade3(ax, s_); ax.set_ylim(0, .9); ax.set_xlim(0, e.L[-1]); ax.set_title(f"{SL[s_]} · {e.nl} layers · {BT[s_]}", loc="left"); ax.set_xlabel("layer"); ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
axes[0].set_ylabel("Jaccard"); axes[3].set_ylabel("Jaccard"); axes[0].legend(fontsize=7, frameon=False, loc="upper left")
fig.suptitle(f"Entanglement by depth for five model sizes · {NDOM} domains × 500 prompts · <bos> excluded · machinery removed · band = layers where top-5,000 sharing is ≥ 80 % of its peak (none if that run spans over half the depth)", x=.01, ha="left", fontsize=9.5)
fig.tight_layout(rect=(0, 0, 1, .93)); fig.savefig(f"{OUT}/act3/a3_mcurve.png", dpi=140); plt.close(fig)
# a3_ladder: the full K ladder for the most entangled pair, the mean, the least entangled pair
fig, axes = plt.subplots(len(SIZES), 3, figsize=(14, 3.1 * len(SIZES)), squeeze=False)
for r, s_ in enumerate(SIZES):
    e = ENT[s_]; n3 = len(e.keep); iu3 = np.triu_indices(n3, 1); avg = np.nanmean([e.J(L, "unbounded") for L in e.L], 0)[iu3]; nm3 = [e.E["names"][i] for i in e.keep]
    hi = np.argmax(avg); lo = np.argmin(avg); pr_hi = (nm3[iu3[0][hi]], nm3[iu3[1][hi]]); pr_lo = (nm3[iu3[0][lo]], nm3[iu3[1][lo]])
    for c, (lab, fn) in enumerate([(f"most entangled: {short(pr_hi[0])} ∩ {short(pr_hi[1])} ({avg[hi]:.2f})", lambda key: e.pairs(key, [pr_hi])[:, 0]), (f"mean over all {NPAIR} pairs (layer-avg {np.nanmean(avg):.2f})", e.allpairs), (f"least entangled: {short(pr_lo[0])} ∩ {short(pr_lo[1])} ({avg[lo]:.2f})", lambda key: e.pairs(key, [pr_lo])[:, 0])]):
        ax = axes[r, c]
        for key in KL3: ax.plot(e.L, fn(key), color=KCOL[key], ls="--" if key == "unbounded" else "-", lw=1.6 if key in ("top5000", "unbounded") else 1.1, label=KLL[key])
        shade3(ax, s_); ax.set_ylim(0, .9); ax.set_xlim(0, e.L[-1]); ax.set_title(f"{SL[s_]} · " + ["most entangled pair", f"mean of {NPAIR} pairs", "least entangled pair"][c], loc="left", fontsize=9); ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        if c != 1 and key == "unbounded": rep.append(f"A3ladder {SL[s_]} {lab}")
        if r == len(SIZES) - 1: ax.set_xlabel("layer")
    axes[r, 0].set_ylabel("Jaccard")
axes[-1, 0].legend(fontsize=7, frameon=False, ncol=2, loc="lower left")
fig.suptitle("The K ladder per size · extreme pairs fixed per size by the layer-averaged all-fired Jaccard · <bos> excluded · machinery removed", x=.01, ha="left", fontsize=9.5)
fig.tight_layout(rect=(0, 0, 1, .97)); fig.savefig(f"{OUT}/act3/a3_ladder.png", dpi=140); plt.close(fig)
# a2_ladder: the same three panels for 27B alone, at a size that reads in the thesis (Section 5.2)
if "27b" in ENT:
    e = ENT["27b"]; n3 = len(e.keep); iu3 = np.triu_indices(n3, 1); avg = np.nanmean([e.J(L, "unbounded") for L in e.L], 0)[iu3]; nm3 = [e.E["names"][i] for i in e.keep]
    hi = np.argmax(avg); lo = np.argmin(avg); pr_hi = (nm3[iu3[0][hi]], nm3[iu3[1][hi]]); pr_lo = (nm3[iu3[0][lo]], nm3[iu3[1][lo]])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for c, (lab, fn) in enumerate([("most entangled pair", lambda key: e.pairs(key, [pr_hi])[:, 0]), (f"mean of {NPAIR} pairs", e.allpairs), ("least entangled pair", lambda key: e.pairs(key, [pr_lo])[:, 0])]):
        ax = axes[c]
        for key in KL3: ax.plot(e.L, fn(key), color=KCOL[key], ls="--" if key == "unbounded" else "-", lw=2 if key in ("top5000", "unbounded") else 1.3, label=KLL[key])
        shade3(ax, "27b"); ax.set_ylim(0, .9); ax.set_xlim(0, e.L[-1]); ax.set_title(lab, loc="left", fontsize=11); ax.set_xlabel("layer")
    axes[0].set_ylabel("Jaccard"); axes[1].legend(fontsize=8.5, frameon=False, ncol=2, loc="upper left")
    rep.append(f"A2ladder 27B most {pr_hi} {avg[hi]:.3f} least {pr_lo} {avg[lo]:.3f}")
    fig.tight_layout(); fig.savefig(f"{OUT}/act2/a2_ladder.png", dpi=150); plt.close(fig)
# a3_cells: same content / same language / neither per size, top-5,000
fig, axes = plt.subplots(len(SIZES), 2, figsize=(10, 2.7 * len(SIZES)), sharey=True, squeeze=False)
for c, s_ in enumerate(SIZES):
    e = ENT[s_]
    for r, (kind, sets) in enumerate(CELLS.items()):
        ax = axes[c, r]
        for lab, prs in sets.items():
            V = e.pairs("top5000", prs); ax.plot(e.L, V.mean(1), color=PAIRCOL[lab], lw=2, label=f"{lab} ({len(prs)} pairs)"); ax.fill_between(e.L, V.min(1), V.max(1), color=PAIRCOL[lab], alpha=.13, lw=0)
            rep.append(f"A3cells {SL[s_]:4s} {kind:9s} {lab:13s} max {V.mean(1).max():.3f} at L{e.L[int(np.argmax(V.mean(1)))]}; band centre / peak {V.mean(1)[BC[s_]]:.3f}; last layer {V.mean(1)[-1]:.3f}")
        shade3(ax, s_); ax.set_ylim(0, .9); ax.set_xlim(0, e.L[-1]); ax.set_title(f"{kind} · {SL[s_]}", loc="left"); ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        if c == len(SIZES) - 1: ax.set_xlabel("layer")
for c in range(len(SIZES)): axes[c, 0].set_ylabel("Jaccard")
for r in range(2): axes[0, r].legend(fontsize=7.5, frameon=False, loc="upper left")
fig.suptitle("Same content vs same language vs neither, per model size · top-5,000 · mean and min–max over pairs · <bos> excluded · machinery removed", x=.01, ha="left", fontsize=9.5)
fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(f"{OUT}/act3/a3_cells.png", dpi=140); plt.close(fig)
# a3_gain: content gain (same content − neither) per K, per size, sentences
fig, axes = plt.subplots(1, len(SIZES), figsize=(3.4 * len(SIZES), 3.4), sharey=True, squeeze=False); axes = axes[0]
SC, NE = CELLS["sentences"]["same content"], CELLS["sentences"]["neither"]
for ax, s_ in zip(axes, SIZES):
    e = ENT[s_]
    for key in KL3:
        g = e.pairs(key, SC).mean(1) - e.pairs(key, NE).mean(1); ax.plot(e.L, g, color=KCOL[key], ls="--" if key == "unbounded" else "-", lw=1.6 if key in ("top5000", "unbounded") else 1.1, label=KLL[key])
        if key in KEYS: rep.append(f"A3gain {SL[s_]:4s} {KL[key]:10s} peak content gain {g.max():.3f} at L{e.L[int(np.argmax(g))]}")
    ax.axhline(0, color="k", lw=.5, ls=":"); shade3(ax, s_); ax.set_ylim(-.05, .5); ax.set_xlim(0, e.L[-1]); ax.set_title(f"{SL[s_]}", loc="left"); ax.set_xlabel("layer"); ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
axes[0].set_ylabel("content gain = same content − neither"); axes[0].legend(fontsize=7, frameon=False, ncol=2, loc="upper left")
fig.suptitle("What the same sentences add over unrelated sentences, per K and size · 10 parallel-language pairs minus 16 unrelated cross-language pairs · <bos> excluded · machinery removed", x=.01, ha="left", fontsize=9.5)
fig.tight_layout(rect=(0, 0, 1, .93)); fig.savefig(f"{OUT}/act3/a3_gain.png", dpi=140); plt.close(fig)
# a3_twoway: content gain vs language gain by relative depth, top-5,000 (solid) and all fired (dashed)
fig, axes = plt.subplots(1, len(SIZES), figsize=(3.4 * len(SIZES), 3.4), sharey=True, squeeze=False); axes = axes[0]
SLG = CELLS["sentences"]["same language"]
for ax, s_ in zip(axes, SIZES):
    e = ENT[s_]; x = np.array(e.L) / (e.nl - 1); pk = {}
    for key, ls in (("top5000", "-"), ("unbounded", "--")):
        cg = e.pairs(key, SC).mean(1) - e.pairs(key, NE).mean(1); lg = e.pairs(key, SLG).mean(1) - e.pairs(key, NE).mean(1)
        ax.plot(x, cg, color=PAIRCOL["same content"], ls=ls, lw=1.8, label=f"content gain · {KL[key]}"); ax.plot(x, lg, color=PAIRCOL["same language"], ls=ls, lw=1.8, label=f"language gain · {KL[key]}")
        pk[key] = (cg - lg).max(); rep.append(f"A3twoway {SL[s_]:4s} {KL[key]:10s} peak content lead (content gain − language gain) {pk[key]:+.3f} at depth {x[int(np.argmax(cg - lg))]:.2f}; language gain min {lg.min():+.3f} at depth {x[int(np.argmin(lg))]:.2f}, last {lg[-1]:+.3f}")
    ax.axhline(0, color="k", lw=.5, ls=":"); ax.grid(alpha=.25); ax.set_ylim(-.1, .5); ax.set_xlim(0, 1); ax.set_title(f"{SL[s_]} ({e.nl} layers) · peak content lead {pk['top5000']:+.2f} / {pk['unbounded']:+.2f}", loc="left", fontsize=8.5); ax.set_xlabel("relative depth")
axes[0].set_ylabel("Jaccard gain over unrelated pairs"); axes[0].legend(fontsize=7, frameon=False, loc="upper left")
fig.suptitle("Content gain (same sentences, other language) vs language gain (other sentences, same language), by relative depth · solid top-5,000, dashed all fired", x=.01, ha="left", fontsize=9.5)
fig.tight_layout(rect=(0, 0, 1, .93)); fig.savefig(f"{OUT}/act3/a3_twoway.png", dpi=140); plt.close(fig)
rep.append("A3 bands (top-5,000 ≥ 80 % of peak, shorter than half the depth): " + ", ".join(f"{SL[s_]} {BT[s_]} of {ENT[s_].nl}" for s_ in SIZES))
# ---------- A2 controls (v2 data only): 16k SAE, split halves, machinery threshold, top-10 %, length-matched same-language ----------
if V2 and "27b" in E3:
    E27 = E3["27b"]; nm = E27["names"]; kp = [i for i, d in enumerate(nm) if d in ix]; jx = {nm[i]: k for k, i in enumerate(kp)}; L27 = sorted(int(l) for l in E27["layers"]); iu27 = np.triu_indices(len(kp), 1)
    def mean_all(E, key, thr="0.6", half=None, keep=None, L_list=None):
        keep = keep if keep is not None else kp; LL = L_list or L27; iu = np.triu_indices(len(keep), 1)
        return np.array([np.nanmean(JM(E, L, key, thr, half)[np.ix_(keep, keep)][iu]) for L in LL])
    # (a) 16k vs 262k at a fixed fraction of what fires (a fixed K is not comparable: 16k fires 5,700–12,300 of its 16,384 features per domain, 262k 9,000–82,000 of 262,144)
    p16 = f"{S}/ent_v4_16k_27b.json"
    if os.path.exists(p16):
        E16 = json.load(open(p16)); k16 = [i for i, d in enumerate(E16["names"]) if d in ix]; L16 = sorted(int(l) for l in E16["layers"])
        fig, ax = plt.subplots(figsize=(8, 4))
        for E_, kk, LL, lab, lw, al in ((E27, kp, L27, "262k", 2, 1), (E16, k16, L16, "16k", 1.3, .6)):
            ax.plot(LL, mean_all(E_, "top10pct", keep=kk, L_list=LL), color="#b45309", lw=lw, alpha=al, label=f"{lab} · top-10 % of what fires")
            ax.plot(LL, mean_all(E_, "unbounded", keep=kk, L_list=LL), color="k", ls="--", lw=lw, alpha=al, label=f"{lab} · all fired")
        ax.plot(L16, mean_all(E16, "top1250", keep=k16, L_list=L16), color="#1d5c8a", lw=1.3, alpha=.6, label="16k · top-1,250 (= 20,000 / 16)")
        for L in (2, 12, 24, 42, 61): rep.append(f"A2ctl 16k L{L:2d} top10pct 262k {mean_all(E27, 'top10pct')[L27.index(L)]:.3f} 16k {mean_all(E16, 'top10pct', keep=k16, L_list=L16)[L16.index(L)]:.3f} | all fired 262k {mean_all(E27, 'unbounded')[L27.index(L)]:.3f} 16k {mean_all(E16, 'unbounded', keep=k16, L_list=L16)[L16.index(L)]:.3f} | 16k top1250 {mean_all(E16, 'top1250', keep=k16, L_list=L16)[L16.index(L)]:.3f} | fired/domain 16k {np.mean([E16['layers'][str(L)]['n_fired'][i] for i in k16]):,.0f}")
        band(ax); ax.set_ylim(0, .9); ax.set_xlim(0, 61); ax.set_xlabel("layer (27B resid_post)"); ax.set_ylabel("Jaccard, mean over all 378 pairs"); ax.legend(ncol=2, fontsize=8, frameon=False, loc="upper left")
        ax.set_title("Control: the 16k SAE beside the 262k SAE · the same 28 domains · machinery removed", loc="left"); fig.tight_layout(); fig.savefig(f"{OUT}/act2/a2_ctl_16k.png", dpi=140); plt.close(fig)
    # (b) split halves + (c) machinery threshold + (d) top-10 %
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for key in KEYS:
        col, ls = KST[key]; h0, h1 = mean_all(E27, key, half=0), mean_all(E27, key, half=1)
        axes[0].plot(L27, h0, color=col, ls=ls, lw=1.6, label=f"{KL[key]} · odd prompts"); axes[0].plot(L27, h1, color=col, ls=ls, lw=1.6, alpha=.45, label=f"{KL[key]} · even prompts")
        rep.append(f"A2ctl split {KL[key]:10s} max |odd − even| over layers {np.abs(h0 - h1).max():.3f}, at L24 {h0[L27.index(24)]:.3f} / {h1[L27.index(24)]:.3f}")
        for thr, a in (("0.5", .35), ("0.6", 1), ("0.8", .6)):
            v = mean_all(E27, key, thr); axes[1].plot(L27, v, color=col, ls=ls, lw=1.8 if thr == "0.6" else 1, alpha=a, label=f"{KL[key]} · machinery ≥ {thr}")
            rep.append(f"A2ctl thr {KL[key]:10s} machinery >= {thr}: L24 {v[L27.index(24)]:.3f} L42 {v[L27.index(42)]:.3f} n_mach L24 {E27['layers']['24']['n_mach'][thr]}")
    v10 = mean_all(E27, "top10pct"); axes[2].plot(L27, mean_all(E27, "top5000"), color=KST["top5000"][0], lw=2, label="top-5,000 (a fixed count)"); axes[2].plot(L27, v10, color="#b45309", lw=2, label="top-10 % of what fires (a fixed fraction)"); axes[2].plot(L27, mean_all(E27, "unbounded"), color="k", ls="--", lw=1.6, label="all fired")
    nf = np.array([np.mean([E27["layers"][str(L)]["n_fired"][i] for i in kp]) for L in L27]); ax2 = axes[2].twinx(); ax2.plot(L27, nf / 10, color="#999", lw=.8, ls=":"); ax2.set_ylabel("features in a domain's top-10 % (dotted grey)", color="#777", fontsize=8); ax2.tick_params(labelsize=7, colors="#777")
    for L in (2, 24, 42, 61): rep.append(f"A2ctl top10pct L{L:2d} J {v10[L27.index(L)]:.3f} (= {nf[L27.index(L)] / 10:,.0f} features) vs top-5,000 {mean_all(E27, 'top5000')[L27.index(L)]:.3f}")
    for ax, t in zip(axes, ("Control: the same M-curve on two halves of the prompts", "Control: the machinery cut at .5 / .6 / .8", "Control: a fixed fraction instead of a fixed count")):
        band(ax); ax.set_ylim(0, .9); ax.set_xlim(0, 61); ax.set_xlabel("layer (27B resid_post)"); ax.set_title(t, loc="left"); ax.legend(fontsize=7, frameon=False, loc="upper left")
    axes[0].set_ylabel("Jaccard, mean over all 378 pairs"); fig.tight_layout(); fig.savefig(f"{OUT}/act2/a2_ctl_robust.png", dpi=140); plt.close(fig)
    # (e) length-matched same-language control (sentences row of the cells figure, with the third line)
    LM = {l: f"lenm_lang_{l}" for l in LANGS}
    if all(LM[l] in nm for l in LANGS):
        jn = {d: i for i, d in enumerate(nm)}
        SETS = {"same content (same sentences, other language)": [(f"par_lang_{a}", f"par_lang_{b}") for i, a in enumerate(LANGS) for b in LANGS[i + 1:]],
                "same language, other texts (full length)": [(f"par_lang_{l}", DIV[l]) for l in LANGS],
                "same language, other texts, cut to the same length": [(f"par_lang_{l}", LM[l]) for l in LANGS],
                "neither (full length)": [(f"par_lang_{a}", DIV[b]) for a in LANGS for b in LANGS if a != b],
                "neither, cut to the same length": [(f"par_lang_{a}", LM[b]) for a in LANGS for b in LANGS if a != b]}
        SCOL = {"same content (same sentences, other language)": "#d62728", "same language, other texts (full length)": "#1f77b4", "same language, other texts, cut to the same length": "#7fb3d5", "neither (full length)": "#7f7f7f", "neither, cut to the same length": "#c0c0c0"}
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        for ax, key in zip(axes, KEYS):
            for lab, prs in SETS.items():
                V = np.array([[JM(E27, L, key)[jn[a], jn[b]] for a, b in prs] for L in L27]); ax.plot(L27, V.mean(1), color=SCOL[lab], lw=2 if "cut" not in lab else 1.6, ls="-" if "cut" not in lab else "--", label=f"{lab} ({len(prs)})")
                for L in (2, 24, 42, 61): rep.append(f"A2ctl lenm {KL[key]:10s} {lab[:44]:44s} L{L:2d} {V[L27.index(L)].mean():.3f}")
            band(ax); ax.set_ylim(0, .9); ax.set_xlim(0, 61); ax.set_xlabel("layer (27B resid_post)"); ax.set_title(f"sentences · {KL[key]}", loc="left")
        axes[0].set_ylabel("Jaccard"); axes[0].legend(fontsize=6.8, frameon=False, loc="upper left")
        fig.suptitle("Control: same language, other texts — full-length Wikipedia vs the same texts cut to the FLORES sentence lengths (row-aligned) · machinery removed", x=.01, ha="left", fontsize=9.5)
        fig.tight_layout(rect=(0, 0, 1, .93)); fig.savefig(f"{OUT}/act2/a2_ctl_length.png", dpi=140); plt.close(fig)
open(f"{S}/v4_story.txt", "w").write("\n".join(rep)); print("\n".join(rep)); print("figs ok")
