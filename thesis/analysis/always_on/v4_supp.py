#!/usr/bin/env python3
"""v4_supp.py — supplementary figure S1: the always-on features across all five model sizes and all 28 domains.
The Act 0 machinery figure (two worked texts, 27B) asked whether the mass spikes are a 27B peculiarity. They are not.
Input  work/supp_machinery.json (supp_machinery.py: machinery = corpus rate >= .6 over the prompts of the 28 domains;
       mass = mean activation per token over tokens 2..n of an average prompt of the domain)
Output figures/supp/s1_machinery_sizes.png, numbers appended to work/v4_story.txt"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../lib"))  # thesis/analysis/lib
from paths import WORK, FIGS
import notitle   # thesis figures carry no header
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = WORK; OUT = f"{FIGS}/supp"
import os; os.makedirs(OUT, exist_ok=True)
D = json.load(open(f"{S}/supp_machinery{os.environ.get('SFX', '_v2')}.json")); SIZES = [s for s in ["270m", "1b", "4b", "12b", "27b"] if s in D]
SL = {"270m": "270M", "1b": "1B", "4b": "4B", "12b": "12B", "27b": "27B"}
LANGS = ["english", "finnish", "french", "german", "swedish"]; CODES = ["python", "javascript", "cpp", "c"]
DIV = {l: (f"div_lang_{l}" if l == "swedish" else f"disjoint_lang_{l}") for l in LANGS}
GROUPS = {"language": [f"par_lang_{l}" for l in LANGS] + [DIV[l] for l in LANGS],
          "code": [f"par_code_{c}" for c in CODES] + [f"div_code_{c}" for c in CODES] + ["code_sql", "code_assembly"],
          "content": ["fact_common", "fact_science", "reasoning_legal", "reasoning_math", "reasoning_moral", "story_fiction", "abstract_philosophy", "abstract_religion"]}
GCOL = {"language": "#1f77b4", "code": "#e08a1e", "content": "#3a9d5d"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 9.5})
rep = []
fig, axes = plt.subplots(2, len(SIZES), figsize=(3.6 * len(SIZES), 6.4), sharey="row", squeeze=False)
for c, s_ in enumerate(SIZES):
    E = D[s_]; names = E["names"]; ix = {d: i for i, d in enumerate(names)}; LAY = sorted(int(l) for l in E["layers"])
    get = lambda key, d: np.array([E["layers"][str(L)][key][ix[d]] for L in LAY])
    for r, key in enumerate(["mach", "top1"]):
        ax = axes[r, c]
        for g, doms in GROUPS.items():
            for d in doms:
                if d in ix: ax.plot(LAY, get(key, d), color=GCOL[g], lw=.8, alpha=.55, label=g if d == doms[0] else None)
        ax.set_ylim(0, 1); ax.set_xlim(0, LAY[-1]); ax.grid(alpha=.25); ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        if r == 1: ax.set_xlabel("layer")
    M = np.array([[E["layers"][str(L)]["mach"][ix[d]] for d in sum(GROUPS.values(), []) if d in ix] for L in LAY])   # (layers, domains)
    T = np.array([[E["layers"][str(L)]["top1"][ix[d]] for d in sum(GROUPS.values(), []) if d in ix] for L in LAY])
    nm = [E["layers"][str(L)]["nmach"] for L in LAY]; k = int(np.argmax(np.median(M, 1))); Lp = LAY[k]
    rp = E["layers"][str(Lp)]; same = len(set(rp["top1_id"])) == 1; tp = float(np.median(T[k]))
    axes[0, c].set_title(f"{SL[s_]} · {E['n_layers']} layers", loc="left")
    axes[0, c].text(.97, .95, f"{min(nm)}–{max(nm):,} always-on features", transform=axes[0, c].transAxes, ha="right", va="top", fontsize=7.5, color="#555")
    axes[1, c].text(.03, .97, f"peak L{Lp}\nalways-on {np.median(M[k]):.2f} of the mass\nloudest one feature {tp:.2f}\n" + (f"= f{rp['top1_id'][0]} in all {len(E['names'])} domains" if same else f"= {len(set(rp['top1_id']))} different features"), transform=axes[1, c].transAxes, ha="left", va="top", fontsize=7.5, color="#333", linespacing=1.45)
    hi = [L for j, L in enumerate(LAY) if np.median(T[j]) >= .5 and len(set(E["layers"][str(L)]["top1_id"])) == 1]
    rep.append(f"S1 {SL[s_]:4s} peak layer L{Lp}: always-on {np.median(M[k]):.2f} of the mass ({nm[k]:,} features), loudest single feature {tp:.2f}, "
               f"{'same feature (f%d) in all %d domains' % (rp['top1_id'][0], len(E['names'])) if same else '%d different loudest features across domains' % len(set(rp['top1_id']))} · "
               f"layers where one feature takes ≥ 50 % in every domain: {['L%d f%d' % (L, E['layers'][str(L)]['top1_id'][0]) for L in hi] or 'none'} · "
               f"always-on features {min(nm)}–{max(nm):,} of 262,144")
axes[0, 0].set_ylabel("share of an average prompt's activation\ncarried by the always-on features")
axes[1, 0].set_ylabel("share carried by the single\nloudest feature of that layer")
h, l = axes[0, 0].get_legend_handles_labels(); fig.legend(h, l, ncol=3, fontsize=8.5, frameon=False, loc="upper left", bbox_to_anchor=(.05, .925))
fig.suptitle("S1 · Always-on features and the activation they carry · five model sizes, all 28 domains, one line per domain\n"
             "always-on = fires in ≥ 60 % of all prompts · mass = activation per token over tokens 2..n of an average prompt", x=.01, ha="left", fontsize=9.5)
fig.tight_layout(rect=(0, 0, 1, .875)); fig.savefig(f"{OUT}/s1_machinery_sizes.png", dpi=140); plt.close(fig)
open(f"{S}/v4_story.txt", "a").write("\n" + "\n".join(rep) + "\n"); print("\n".join(rep)); print("supp ok")

# ---------- S2: the same thing as a map — every model, every layer, where the peaks sit ----------
from matplotlib.colors import LinearSegmentedColormap
CM = LinearSegmentedColormap.from_list("mach", ["#ffffff", "#fde8c8", "#e08a1e", "#8c3a06", "#3a1702"]); CM.set_bad("#f0f0f0")
NL = {s_: D[s_]["n_layers"] for s_ in SIZES}; W = max(NL.values())
fig, axes = plt.subplots(2, 1, figsize=(15, 6.4), sharex=True)
for r, (key, lab) in enumerate([("top1", "the single loudest feature of the layer"), ("mach", "all always-on features together")]):
    A = np.full((len(SIZES), W), np.nan); ID = {}
    for i, s_ in enumerate(SIZES):
        E = D[s_]
        for L in sorted(int(l) for l in E["layers"]):
            v = np.array(E["layers"][str(L)][key]); A[i, L] = np.median(v)
            if key == "top1" and np.median(v) >= .25 and len(set(E["layers"][str(L)]["top1_id"])) == 1: ID[(i, L)] = E["layers"][str(L)]["top1_id"][0]
    ax = axes[r]; im = ax.imshow(A, cmap=CM, vmin=0, vmax=1, aspect="auto")
    for i in range(len(SIZES)):
        for L in range(W):
            v = A[i, L]
            if not np.isnan(v) and v >= .25:
                ax.text(L, i - (.16 if (i, L) in ID else 0), f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="white" if v > .55 else "#1a1a1a")
                if (i, L) in ID: ax.text(L, i + .22, f"f{ID[(i, L)]}", ha="center", va="center", fontsize=5.8, color="white" if v > .55 else "#5a3a12")
    ax.set_yticks(range(len(SIZES))); ax.set_yticklabels([f"{SL[s_]} · {NL[s_]} layers" for s_ in SIZES], fontsize=9)
    ax.set_xticks(range(0, W, 2)); ax.set_xticklabels(range(0, W, 2), fontsize=7); ax.set_title(f"share of an average prompt's activation carried by {lab}  (median over the 28 domains)", loc="left", fontsize=9.5)
    for i, s_ in enumerate(SIZES): ax.add_patch(plt.Rectangle((-.5, i - .5), NL[s_], 1, fill=False, ec="#999", lw=.6))
    ax.set_xlim(-.5, W - .5)
axes[1].set_xlabel("layer (each model starts at its own layer 0; grey = the model has no such layer)")
lines = []
for s_ in SIZES:
    E = D[s_]; LAY = sorted(int(l) for l in E["layers"])
    pk = [(L, float(np.median(E["layers"][str(L)]["top1"]))) for L in LAY if np.median(E["layers"][str(L)]["top1"]) >= .25]
    lines.append(f"{SL[s_]}: " + (", ".join(f"L{L} ({L / (NL[s_] - 1):.2f} of the depth, {v:.2f})" for L, v in pk) if pk else "no single-feature peak"))
DEP = {s_: [L / (NL[s_] - 1) for L in sorted(int(l) for l in D[s_]["layers"]) if np.median(D[s_]["layers"][str(L)]["top1"]) >= .25] for s_ in SIZES}
fig.suptitle("S2 · Where the always-on peaks sit · Gemma 3, five sizes, every layer · 28 domains · always-on = fires in ≥ 60 % of all prompts\n"
             "A cell's label is the feature id: at those layers one feature is the loudest in every domain. In depth: " +
             "; ".join(f"{SL[s_]} " + (", ".join(f"{d:.2f}" for d in DEP[s_]) if DEP[s_] else "none") for s_ in SIZES), x=.008, y=.985, ha="left", va="top", fontsize=8.5)
fig.tight_layout(rect=(0, 0, .915, .90)); fig.colorbar(im, cax=fig.add_axes([.928, .12, .011, .62]), label="share of the activation mass"); fig.savefig(f"{OUT}/s2_machinery_map.png", dpi=150); plt.close(fig)
open(f"{S}/v4_story.txt", "a").write("\n".join("S2 " + l for l in lines) + "\n"); print("\n".join("S2 " + l for l in lines)); print("s2 ok")
