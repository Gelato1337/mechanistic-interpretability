#!/usr/bin/env python3
"""tf_train_export.py — training table for bridge_v4.py: work/tf_train.json {"stats": [...], "rows": [{"id", "y", "x", "src"}]}
from the Sonnet labels (v3_labels.json, resolved cards of the five classes) with their TF statistics (tf_labelled.json, tf_extract.py)
and, when present, the Qwen labels of the domain-wide sample (env TAG, default 2: qwen_dom{TAG}_labels.jsonl + tf_dom{TAG}.json; resolved = class in the five
and confidence != low). 'global' and unresolved cards are not training rows."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../lib"))  # thesis/analysis/lib
from paths import WORK
import json, os, collections
S = WORK; CLS = ["position", "token", "word", "concept", "context"]
LAB = json.load(open(f"{S}/v3_labels.json")); TF = json.load(open(f"{S}/tf_labelled.json"))
STATS = [s for s in next(iter(TF.values())) if s not in ("topstr", "topword")] + ["layer"]
rows = []
for k, v in LAB.items():
    if v["resolved"] and v["class"] in CLS and k in TF and TF[k]["nf"] is not None:
        x = {s: TF[k][s] for s in STATS if s != "layer"}; x["layer"] = v["layer"]; rows.append({"id": k, "y": v["class"], "x": x, "src": "sonnet"})
TAG = os.environ.get("TAG", "2")   # domain-wide sample version: qwen_dom{TAG}_labels.jsonl + tf_dom{TAG}.json + dom_cards{TAG}/index.json
if os.path.exists(f"{S}/qwen_dom{TAG}_labels.jsonl") and os.path.exists(f"{S}/tf_dom{TAG}.json"):
    TD = json.load(open(f"{S}/tf_dom{TAG}.json")); IDX = json.load(open(f"{S}/dom_cards{TAG}/index.json"))
    for line in open(f"{S}/qwen_dom{TAG}_labels.jsonl"):
        r = json.loads(line); k = r["id"]
        if r.get("class") in CLS and r.get("confidence") != "low" and k in TD and TD[k]["nf"] is not None:
            x = {s: TD[k][s] for s in STATS if s != "layer"}; x["layer"] = IDX[k]["layer"]; rows.append({"id": k, "y": r["class"], "x": x, "src": "qwen_dom"})
json.dump({"stats": STATS, "rows": rows}, open(f"{S}/tf_train.json", "w"))
print("rows", len(rows), dict(collections.Counter(r["src"] for r in rows)), dict(collections.Counter(r["y"] for r in rows)), "stats", len(STATS))
