#!/usr/bin/env python3
"""tf_extract.py (CPU) — cut the rows of the judged features out of work/tf_feats/L{L}.npz (tf_feats.py) into
work/tf_labelled.json {id: {stat: value, ..., "topstr": str}} for ids of work/v3_labels_ids.json ([id...],
id = <ex>_L<layer>_f<feature>; default: the keys of work/v3_labels.json). Env IDS (a JSON list of ids, or a dict keyed by id) / OUT
override the files; the domain-wide sample: IDS=work/dom_cards2/index.json OUT=work/tf_dom2.json."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../lib"))  # thesis/analysis/lib
from paths import WORK
import json, re, os, collections, numpy as np
S = WORK
IDS = os.environ.get("IDS", f"{S}/v3_labels_ids.json"); OUTF = os.environ.get("OUT", f"{S}/tf_labelled.json")   # env IDS / OUT: other id lists (dom sample)
ids = json.load(open(IDS)) if os.path.exists(IDS) else json.load(open(f"{S}/v3_labels.json"))   # a list, or a dict whose keys are the ids
ids = list(ids); by = collections.defaultdict(list)
for k in ids:
    m = re.match(r".*_L(\d+)_f(\d+)$", k); by[int(m.group(1))].append((k, int(m.group(2))))
out = {}
for L in sorted(by):
    z = np.load(f"{S}/tf_feats/L{L}.npz", allow_pickle=True); keys = [k for k in z.files if k not in ("topstr", "topword", "names", "qty", "MASS", "NFD")]; A = {k: z[k] for k in keys}; ts = z["topstr"]; tw = z["topword"]
    for k, f in by[L]:
        out[k] = {s: (None if np.isnan(A[s][f]) else float(A[s][f])) for s in keys}; out[k]["topstr"] = str(ts[f]); out[k]["topword"] = str(tw[f])
    print(L, len(by[L]), flush=True)
json.dump(out, open(OUTF, "w")); print("TF_EXTRACT_DONE", len(out))
