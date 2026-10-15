#!/usr/bin/env python3
"""dom_cards.py — judge cards (plain text) for the domain-wide sample: work/dom_dump{TAG}.json (dom_dump.py; env TAG, e.g. TAG=2) ->
work/dom_cards/dom_L<layer>_<part>.txt (<= CHUNK cards per file) + dom_cards/index.json.
Same card layout as pilot_ex_cards.py; the example texts are the feature's two loudest passages of its home domain."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../lib"))  # thesis/analysis/lib
from paths import WORK
import json, os, collections
S = WORK; TAG = os.environ.get("TAG", ""); P = json.load(open(f"{S}/dom_dump{TAG}.json")); CHUNK = 20
D = f"{S}/dom_cards{TAG}"; os.makedirs(D, exist_ok=True)
def mark(toks, hits):
    s = ""
    for i, t in enumerate(toks):
        t = t.replace("\n", "\\n"); s += f"[[{t}|{hits[i]:.0f}]]" if i in hits else t
    return s
def card(cid, v):
    L, f, fa, d = v["layer"], v["f"], v["facts"], v["domain"]; hum = fa["corpus_rate"] >= 0.6; x = fa["domains"][d]
    h = [f"=== card {cid}",
         f"FACTS  layer {L}, feature {f}. Sampled from domain {d}: rank {v['rank']+1} of {v['n']} features firing in that domain by activation mass.",
         f"  Corpus (500 prompts x 33 domains): fires in {100*fa['corpus_rate']:.1f}% of prompts" + ("  <<ALWAYS-ON: fires in >=60% of all prompts>>" if hum else "") + ".",
         f"  Home domain {d}: {100*x['rate']:.0f}% of prompts, {100*x['cov']:.0f}% of tokens when it fires, {100*x['first']:.0f}% of fires on the first token.",
         f"  Across all its fires: {fa['distinct_tokens']} distinct token strings; {100*fa['word_cont']:.0f}% of fires are on a token that continues a word (no leading space, alphabetic).",
         f"  Most frequent tokens among its fires (together {100*fa['top_share']:.0f}% of all fires): " + ", ".join(f"{t!r} {100*c:.0f}%" for t, c in fa["top_tokens"]),
         "EXAMPLE TEXTS  (the two passages of the home domain where it peaks highest; fires marked [[token|activation]])"]
    for dd, j, toks, hits in v["texts"]: h.append(f"  {dd} prompt {j}: " + mark(toks, {int(k): a for k, a in hits.items()}))
    if v["other"]:
        dd, j, toks, hits = v["other"]; h.append(f"OTHER PASSAGE  ({dd} prompt {j})"); h.append("  " + mark(toks, {int(k): a for k, a in hits.items()}))
    for grp, lab in (("lang", "LANGUAGE"), ("code", "CODE")):
        rows = v["random"][grp]
        h.append(f"RANDOM FIRES IN {lab} PROMPTS  ({len(rows)} random prompts where it fires; strongest fire with 3 tokens of context each side)")
        for dd, j, lo, toks, hits in rows: h.append(f"  {dd}: ..." + mark(toks, {int(k): a for k, a in hits.items()}) + "...")
        if not rows: h.append("  (none)")
    return "\n".join(h), hum
index = {}; nfile = 0; nchar = 0
for L, rec in P.items():
    cards = sorted(rec.items(), key=lambda kv: (kv[1]["domain"], kv[1]["rank"]))
    for part in range(0, len(cards), CHUNK):
        name = f"dom_L{L}_{part // CHUNK}"; txt = []
        for cid, v in cards[part:part + CHUNK]:
            t, hum = card(cid, v); txt.append(t); nchar += len(t)
            index[cid] = {"domain": v["domain"], "layer": int(L), "f": v["f"], "rank": v["rank"], "stratum": v["stratum"], "mass": v["mass"], "hum": hum, "corpus_rate": v["facts"]["corpus_rate"], "file": name}
        open(f"{D}/{name}.txt", "w").write("\n\n".join(txt) + "\n"); nfile += 1
json.dump(index, open(f"{D}/index.json", "w"), indent=1)
print("cards", len(index), "files", nfile, "chars/card", nchar // max(1, len(index)), dict(collections.Counter(v["stratum"] for v in index.values())))
