#!/usr/bin/env python3
"""
make_lenmatched.py — the length-matched control of the thesis: for each language, diverse-tier prompt i
(disjoint_lang_{en,fi,fr,de,sv}; --swedish div_lang_swedish for the thesis data) cut to exactly the Gemma token length of FLORES prompt i of the
same language (rows in prompt_id order). Writes micro_domain lenm_lang_{language}, prompt_id from 20000.

    python thesis/data/make_lenmatched.py --prompts thesis/data/prompts.csv --out thesis/data/prompts_lenmatched.csv
"""
import argparse, csv, collections
import numpy as np
from transformers import AutoTokenizer

ap = argparse.ArgumentParser(); ap.add_argument("--prompts", nargs="+", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--tokenizer", default="google/gemma-3-27b-pt"); ap.add_argument("--swedish", default=None, help="name of the Swedish diverse domain")
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.tokenizer)
by = collections.defaultdict(list)
for f in a.prompts:
    for r in csv.DictReader(open(f)):
        by[r["micro_domain"]].append((int(r["prompt_id"]), r["text"]))
LANGS = ["english", "finnish", "french", "german", "swedish"]; DIV = {l: f"disjoint_lang_{l}" for l in LANGS}
if a.swedish: DIV["swedish"] = a.swedish
out, pid = [], 20000
for l in LANGS:
    par, div = sorted(by[f"par_lang_{l}"]), sorted(by[DIV[l]]); lens = []
    for (_, tp), (_, td) in zip(par, div):
        n = len(tok(tp, truncation=True, max_length=512)["input_ids"]); ids = tok(td, truncation=True, max_length=512)["input_ids"]
        cut = tok.decode(ids[1:n], skip_special_tokens=True)
        out.append({"prompt_id": pid, "micro_domain": f"lenm_lang_{l}", "text": cut}); pid += 1; lens.append((n, len(tok(cut)["input_ids"])))
    x = np.array(lens); print(l, "FLORES tokens", x[:, 0].mean().round(1), "cut tokens", x[:, 1].mean().round(1), "exact", (x[:, 0] == x[:, 1]).mean().round(3))
w = csv.DictWriter(open(a.out, "w", newline=""), fieldnames=["prompt_id", "micro_domain", "text"]); w.writeheader(); w.writerows(out)
