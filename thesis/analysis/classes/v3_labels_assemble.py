#!/usr/bin/env python3
"""v3_labels_assemble.py — work/v3_labels.json = v2_labels.json (mass-run labels: shared, machinery, pop) + the judge
labels of the peak-only cards (pilot_cards_v3/labels_v3_{k}.jsonl, index.json). Same entry layout as v2: {track, example, bucket,
layer, class, resolved, what}; resolved = class not other/None and confidence != low. Prints per-track counts and the missing ids."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../lib"))  # thesis/analysis/lib
from paths import WORK
import json, glob, collections
S = WORK
LAB = json.load(open(f"{S}/v2_labels.json")); IDX = json.load(open(f"{S}/pilot_cards_v3/index.json"))
got = {}
for fn in glob.glob(f"{S}/pilot_cards_v3/labels_v3_*.jsonl"):
    for line in open(fn):
        line = line.strip()
        if not line: continue
        try: r = json.loads(line)
        except json.JSONDecodeError: print("bad line", fn, line[:80]); continue
        got[r["id"]] = r
missing = [k for k in IDX if k not in got]; new = 0
for k, v in IDX.items():
    if k not in got: continue
    r = got[k]; cls = r.get("class"); conf = r.get("confidence")
    LAB[k] = {"track": "shared", "example": v["example"], "bucket": v["bucket"], "layer": v["layer"], "class": cls,
              "resolved": cls not in (None, "other") and conf != "low", "what": r.get("what", ""), "rank": "peak"}; new += 1
json.dump(LAB, open(f"{S}/v3_labels.json", "w"), indent=0)
print("v3 cards", len(IDX), "labelled", len(got), "added", new, "missing", len(missing), missing[:20])
print("classes of the new cards:", dict(collections.Counter((LAB[k]["class"] if LAB[k]["resolved"] else "unresolved") for k in IDX if k in got)))
