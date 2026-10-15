#!/usr/bin/env python3
"""v2_ex_sel.py — what the judge sees: the selection file for pilot_ex_dump.py, built from work/heur_shared.json.
(a) Worked examples, all 62 layers: opera = par_lang prompt 21, prime = par_code prompt 28. Per layer and example:
    track "shared"    = the 10 loudest non-machinery shared features by mass over tokens 2..n whose heuristic bucket is
                        undetermined or wordcand (token / position / rare are settled by the heuristic and not judged);
                        the full top-10 with buckets is kept in "top10" for the figures
    track "machinery" = the 10 loudest shared-machinery features (rate >= .6) with coverage < .9 (the constant features are
                        excluded by rule; the first-token sinks have no mass over tokens 2..n and rank last)
(b) Population sample at the snapshot layers SNAP: pseudo-examples lang_pJ / code_pJ (prompt set J, prompts 0-49 minus 21/28).
    Per layer and group: N_UND undetermined + N_WC wordcand rows drawn at random over sets (distinct features), and heuristic
    controls N_CTL each of token and position so the heuristic's precision is measured by the same judge.
Output work/pilot_ex_sel.json  {L: {example: {domains, prompt, features: [{f, track, bucket, mass, rank, nshared, cov, corpus_rate, loud}]}}}
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../lib"))  # thesis/analysis/lib
from paths import WORK
import json, collections, numpy as np
S = WORK
H = json.load(open(f"{S}/heur_shared.json")); F = H["facts"]
G = {"lang": ["par_lang_english", "par_lang_finnish", "par_lang_french", "par_lang_german", "par_lang_swedish"],
     "code": ["par_code_python", "par_code_javascript", "par_code_cpp", "par_code_c"]}
EX = {"opera": ("lang", 21), "prime": ("code", 28)}
SNAP = [0, 6, 12, 17, 24, 34, 42, 52, 61]; N_UND, N_WC, N_CTL = 50, 20, 10
rng = np.random.default_rng(2026)
LAY = sorted(int(k) for k in H["sets"]["lang"]); sel = {}; top10 = {}
def row(r, L, track, rank, nsh):
    f, sc, cov, b, loud = r; fa = F[str(L)].get(str(f), {})
    return {"f": f, "track": track, "bucket": b, "mass": sc, "rank": rank, "nshared": nsh, "cov": cov, "corpus_rate": fa.get("rate", 0), "loud": loud}
for L in LAY:
    sel[str(L)] = {}
    for ex, (g, j) in EX.items():
        rows = H["sets"][g][str(L)][j]                       # sorted by mass desc
        shared = [r for r in rows if r[3] != "machinery"]; mach = [r for r in rows if r[3] == "machinery" and r[1] > 0 and r[2] < 0.9]
        feats = [row(r, L, "shared", i + 1, len(shared)) for i, r in enumerate(shared[:10])]
        top10.setdefault(ex, {})[L] = feats
        judged = [x for x in feats if x["bucket"] in ("undetermined", "wordcand")]
        judged += [row(r, L, "machinery", i + 1, len(mach)) for i, r in enumerate(mach[:10])]
        sel[str(L)][ex] = {"domains": G[g], "prompt": j, "features": judged}
    if L in SNAP:
        for g in G:
            pool = collections.defaultdict(list)
            for j, rows in enumerate(H["sets"][g][str(L)]):
                if j in (21, 28): continue
                for i, r in enumerate(rows):
                    pool[r[3]].append((j, i, r, len([q for q in rows if q[3] != "machinery"])))
            for b, n in (("undetermined", N_UND), ("wordcand", N_WC), ("token", N_CTL), ("position", N_CTL)):
                cand = pool[b]; seen = set(); pick = []
                for k in rng.permutation(len(cand)):
                    j, i, r, nsh = cand[k]
                    if r[0] in seen: continue
                    seen.add(r[0]); pick.append((j, i, r, nsh))
                    if len(pick) == n: break
                for j, i, r, nsh in pick:
                    e = sel[str(L)].setdefault(f"{g}_p{j}", {"domains": G[g], "prompt": j, "features": []})
                    e["features"].append(row(r, L, "pop", i + 1, nsh))
json.dump(sel, open(f"{S}/pilot_ex_sel.json", "w")); json.dump(top10, open(f"{S}/v2_ex_top10.json", "w"))
n = collections.Counter((e["track"] if e["track"] != "pop" else f"pop_{e['bucket']}") for L in sel for ex in sel[L].values() for e in ex["features"])
print("cards:", dict(n), "total", sum(n.values()), "| pseudo-examples", sum(1 for L in sel for k in sel[L] if "_p" in k))
