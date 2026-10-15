#!/usr/bin/env python3
"""pilot_ex_cards.py — turn work/pilot_ex_dump.json into judge cards (plain text).
One card per (example, layer, feature): a FACTS header of machine-computed numbers, the example's own texts with the feature's
fires marked as [[token|act]], one more passage from a member domain, and random fires from other corpus prompts split into
language and code prompts. Shared machinery (corpus fire rate >= .6) is printed as a flag, not decided by the judge. The
heuristic bucket is NOT printed (the token / position controls measure the heuristic against the judge).
Tracks: shared = worked-example top-10 (opera / prime, all layers); machinery = their shared-machinery top-10; pop = the
population sample at the snapshot layers (pseudo-examples lang_pJ / code_pJ = prompt set J).
Output work/pilot_cards/<track>_L<layer>_<part>.txt (<= CHUNK cards per file) and pilot_cards/index.json.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../lib"))  # thesis/analysis/lib
from paths import WORK
import json, os, collections
S = WORK
P = json.load(open(f"{S}/pilot_ex_dump.json")); CHUNK = 20
os.makedirs(f"{S}/pilot_cards", exist_ok=True)
def mark(toks, hits):
    s = ""
    for i, t in enumerate(toks):
        t = t.replace("\n", "\\n")
        s += f"[[{t}|{hits[i]:.0f}]]" if i in hits else t
    return s
def card(v, texts, L):
    ex, f, fa, track = v["example"], v["f"], v["facts"], v["track"]; doms = list(texts.keys()); hum = fa["corpus_rate"] >= 0.6
    what = {"shared": " (shared-machinery features excluded).", "machinery": " (among the shared-machinery features).",
            "pop": " (shared-machinery features excluded)."}[track]
    h = [f"=== card {ex}_L{L}_f{f}",
         f"FACTS  layer {L}, feature {f}. Fires in all {len(doms)} texts of the example; rank {v['rank']} of {v['nshared']} such features by activation mass" + what,
         f"  In the example it fires on {100*v['cov']:.0f}% of the tokens (mean over the texts).",
         f"  Corpus (500 prompts x 27 domains): fires in {100*fa['corpus_rate']:.1f}% of prompts"
         + ("  <<ALWAYS-ON: fires in >=60% of all prompts>>" if hum else "") + ".",
         "  Member domains: " + "; ".join(f"{d.replace('par_','')} {100*x['rate']:.0f}% of prompts, {100*x['cov']:.0f}% of tokens when it fires, {100*x['first']:.0f}% of fires on the first token"
                                            for d, x in fa["domains"].items()),
         f"  Across all its fires: {fa['distinct_tokens']} distinct token strings; {100*fa['word_cont']:.0f}% of fires are on a token that continues a word (no leading space, alphabetic).",
         f"  Most frequent tokens among its fires (together {100*fa['top_share']:.0f}% of all fires): " + ", ".join(f"{t!r} {100*c:.0f}%" for t, c in fa["top_tokens"]),
         "EXAMPLE TEXTS  (fires marked [[token|activation]])"]
    for d, (toks, hits) in texts.items():
        h.append(f"  {d}: " + mark(toks, {int(k): a for k, a in hits[str(f)].items()}))
    if v["other"]:
        d, j, toks, hits = v["other"]
        h.append(f"OTHER PASSAGE  ({d} prompt {j})"); h.append("  " + mark(toks, {int(k): a for k, a in hits.items()}))
    for grp, lab in (("lang", "LANGUAGE"), ("code", "CODE")):
        rows = v["random"][grp]
        h.append(f"RANDOM FIRES IN {lab} PROMPTS  ({len(rows)} random prompts where it fires; strongest fire with 3 tokens of context each side)")
        for d, j, lo, toks, hits in rows:
            h.append(f"  {d}: ..." + mark(toks, {int(k): a for k, a in hits.items()}) + "...")
        if not rows: h.append("  (none)")
    return "\n".join(h), hum
index = {}; nfile = collections.Counter()
for L, rec in P.items():
    by = collections.defaultdict(list)
    for v in rec["cards"].values(): by[v["track"]].append(v)
    for track, cards in by.items():
        cards.sort(key=lambda v: (v["example"], v["rank"]))
        for part in range(0, len(cards), CHUNK):
            name = f"{track}_L{L}_{part // CHUNK}"; txt = []
            for v in cards[part:part + CHUNK]:
                t, hum = card(v, rec["texts"][v["example"]], L); txt.append(t)
                index[f"{v['example']}_L{L}_f{v['f']}"] = {"example": v["example"], "track": track, "bucket": v["bucket"], "layer": int(L), "f": v["f"], "rank": v["rank"],
                                                          "mass": v["mass"], "cov": v["cov"], "hum": hum, "corpus_rate": v["facts"]["corpus_rate"], "file": name}
            open(f"{S}/pilot_cards/{name}.txt", "w").write("\n\n".join(txt) + "\n"); nfile[track] += 1
json.dump(index, open(f"{S}/pilot_cards/index.json", "w"), indent=1)
print("cards", collections.Counter(v["track"] for v in index.values()), "files", dict(nfile))
