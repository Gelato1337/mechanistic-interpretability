#!/usr/bin/env python3
"""agg_src.py — one loader for the per-domain feature statistics of every size and SAE width:
$SAE_STATS/{width}/{size}[_{tag}]/L{L}.npz written by scripts/aggregate.py (JumpReLU z = pre * 1[pre > theta],
<bos> excluded, the 14 philosophy/religion duplicates dropped, odd/even prompt halves kept).
load(size, L, width, extra=tag) returns dict names, MASS, NFD, NF1, NPR, nf, rate, NFD_h, MASS_h, NPR_h (+ the extra domains' rows
appended when extra is given; nf / rate stay those of the 28 domains = the corpus)."""
from paths import WORK, stats_file, DIVERSE_LANG
import numpy as np
S = WORK   # intermediate results of the analysis scripts
LANGS = ["english", "finnish", "french", "german", "swedish"]; CODES = ["python", "javascript", "cpp", "c"]
DOMAINS = [f"par_lang_{l}" for l in LANGS] + DIVERSE_LANG + [f"par_code_{c}" for c in CODES] + [f"div_code_{c}" for c in CODES] + \
          ["code_sql", "code_assembly", "reasoning_math", "reasoning_moral", "reasoning_legal", "abstract_philosophy", "abstract_religion", "fact_science", "fact_common", "story_fiction"]
assert len(DOMAINS) == 28
def load(size, L, width="262k", extra=None):
    """The 28 domains of the thesis, in the file's (sorted) order; nf and rate = the corpus of exactly these domains (13,986 prompts)."""
    z = np.load(stats_file(size, L, width), allow_pickle=True); d = {k: z[k] for k in z.files}; d["names"] = [str(x) for x in d["names"]]
    sel = [i for i, n in enumerate(d["names"]) if n in DOMAINS]; assert len(sel) == 28; d["names"] = [d["names"][i] for i in sel]
    for k in ("MASS", "NFD", "NF1", "NPR"): d[k] = d[k][sel]
    for k in ("MASS_h", "NFD_h", "NPR_h"): d[k] = d[k][:, sel]
    d["nf"] = d["NFD"].sum(0); d["rate"] = (d["nf"] / d["NPR"].sum()).astype(np.float32)
    if extra:
        e = np.load(stats_file(size, L, width, extra), allow_pickle=True)
        d["names"] = d["names"] + [str(x) for x in e["names"]]
        for k in ("MASS", "NFD", "NF1", "NPR"): d[k] = np.concatenate([d[k], e[k]], 0)
        for k in ("MASS_h", "NFD_h", "NPR_h"): d[k] = np.concatenate([d[k], e[k]], 1)
    return d
def base_rows(d, names): return [d["names"].index(n) for n in names]   # reorder to a caller's name order
