#!/usr/bin/env python3
"""dataset_stats.py (CPU) — per micro-domain prompt counts and Gemma token lengths of the prompt file(s) (paths.PROMPTS),
tokenised as the captures tokenise (google/gemma-3-27b-pt tokenizer,
<bos> included, 512-token cap). Output work/dataset_stats.json {domain: {n, chars_mean, tok_mean, tok_median, tok_min, tok_max, tok_capped}}."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../analysis/lib"))  # thesis/analysis/lib
from paths import WORK, PROMPTS
import os, json, csv, collections, numpy as np
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("google/gemma-3-27b-pt")
rows = [r for f in PROMPTS for r in csv.DictReader(open(f, newline=""))]
by = collections.defaultdict(list)
for r in rows: by[r["micro_domain"]].append(r["text"])
out = {}
for d, texts in sorted(by.items()):
    n = [len(tok(t, truncation=True, max_length=512)["input_ids"]) for t in texts]; n = np.array(n)
    out[d] = {"n": len(texts), "chars_mean": round(float(np.mean([len(t) for t in texts])), 1), "tok_mean": round(float(n.mean()), 1), "tok_median": int(np.median(n)), "tok_min": int(n.min()), "tok_max": int(n.max()), "tok_capped": int((n >= 512).sum())}
    print(d, out[d], flush=True)
json.dump(out, open(f"{WORK}/dataset_stats.json", "w"), indent=1); print("DATASET_STATS_DONE")
