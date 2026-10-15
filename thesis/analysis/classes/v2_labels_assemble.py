#!/usr/bin/env python3
"""v2_labels_assemble.py — collect the judge's answers (work/pilot_cards/labels_<file>.jsonl, one per card file) into
work/v2_labels.json {card id: {class, what, confidence, track, bucket, layer, example, f, rank, mass, cov, corpus_rate,
resolved}} and print (a) coverage: cards with / without an answer per track, (b) the class split per track and layer,
(c) heuristic precision: judge class on the token / position control cards of the population sample.
Rule: class "other" or confidence "low" -> resolved = False (drawn as unresolved, never folded into a class).
Judge classes: position / token / word / concept / context / other (judge/protocol_cards.txt); machinery: position / token / global / other."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../lib"))  # thesis/analysis/lib
from paths import WORK
import json, glob, collections, os
S = WORK; C = f"{S}/pilot_cards"
idx = json.load(open(f"{C}/index.json")); lab = {}
for p in glob.glob(f"{C}/labels_*.jsonl"):
    for line in open(p):
        line = line.strip()
        if not line.startswith("{"): continue
        try: r = json.loads(line)
        except json.JSONDecodeError: print("bad line", p, line[:80]); continue
        if r.get("id") in idx: lab[r["id"]] = r
out = {}; miss = collections.Counter(); have = collections.Counter()
for cid, m in idx.items():
    if cid not in lab: miss[m["track"]] += 1; continue
    r = lab[cid]; have[m["track"]] += 1
    out[cid] = dict(m, **{"class": r.get("class", "other"), "what": r.get("what", ""), "confidence": r.get("confidence", "low"),
                          "resolved": r.get("class") not in (None, "other") and r.get("confidence") != "low"})
json.dump(out, open(f"{S}/v2_labels.json", "w"), indent=0)
print("answered", dict(have), "| missing", dict(miss))
for track in ("shared", "machinery", "pop"):
    byL = collections.defaultdict(collections.Counter)
    for v in out.values():
        if v["track"] == track: byL[v["layer"]][v["class"] if v["resolved"] else "unresolved"] += 1
    print(f"--- {track}")
    for L in sorted(byL): print(f"  L{L:2d}", dict(byL[L]))
print("--- controls (population sample): heuristic bucket -> judge class")
ctl = collections.defaultdict(collections.Counter)
for v in out.values():
    if v["track"] == "pop": ctl[v["bucket"]][v["class"] if v["resolved"] else "unresolved"] += 1
for b in ("token", "position", "wordcand", "undetermined"): print(f"  {b:12s}", dict(ctl[b]))
