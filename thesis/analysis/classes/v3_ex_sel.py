#!/usr/bin/env python3
"""v3_ex_sel.py — worked-example top-10 by PEAK (the v1 rank) for story_figs_v3.
Per layer and example (opera = par_lang prompt 21, prime = par_code prompt 28), from work/tokdump_all.json (27B resid_post,
all 262k features, nothing filtered, <bos> excluded): shared = fires on >= 1 token of every text; shared machinery (corpus rate
>= .6 over 500 x 33 prompts, from heur_shared.json) counted, not ranked; the 10 loudest of the rest by
    peak = min over the texts of the feature's max activation in that text   (v1 rule, fig_snap.py)
Heuristic bucket, cov, mass, loud come from the heur_shared.json row of the same feature (same capture, same rules).
Output work/v3_ex_top10.json  {ex: {L: [{f, bucket, peak, mass, rank, nshared, cov, corpus_rate, loud}]}} in the field layout of
v2_ex_top10.json (+ peak), and work/v3_judge_needed.json = the top-10 features bucketed undetermined / wordcand that have no
label in v2_labels.json (the cards the judge still has to see)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../lib"))  # thesis/analysis/lib
from paths import WORK
import json, os, collections
S = WORK
D = json.load(open(f"{S}/tokdump_all.json")); H = json.load(open(f"{S}/heur_shared.json")); F = H["facts"]
LAB = json.load(open(f"{S}/v2_labels.json")) if os.path.exists(f"{S}/v2_labels.json") else {}
EX = {"opera": ("lang", 21), "prime": ("code", 28)}; N = 10
top = {}; need = {}; log = collections.Counter()
for ex, (g, j) in EX.items():
    doms = D["examples"][ex]["domains"]; top[ex] = {}
    for L in sorted(int(k) for k in D["layers"]):
        T = D["layers"][str(L)]["texts"]; mx = {d: collections.defaultdict(float) for d in doms}
        for d in doms:
            for t in T[d][1:]:
                for f, a in t["f"]: mx[d][f] = max(mx[d][f], a)
        sh = set.intersection(*[set(mx[d]) for d in doms])
        rowmap = {r[0]: r for r in H["sets"][g][str(L)][j]}                     # f -> [f, mass, cov, bucket, loud]
        assert sh == set(rowmap), (ex, L, len(sh), len(rowmap))                 # same capture, same shared set
        rest = [f for f in sh if rowmap[f][3] != "machinery"]
        rest.sort(key=lambda f: -min(mx[d][f] for d in doms))
        rows = []
        for i, f in enumerate(rest[:N]):
            _, mass, cov, b, loud = rowmap[f]; fa = F[str(L)].get(str(f), {})
            rows.append({"f": f, "bucket": b, "peak": min(mx[d][f] for d in doms), "mass": mass, "rank": i + 1, "nshared": len(rest),
                         "cov": cov, "corpus_rate": fa.get("rate", 0), "loud": loud})
            if b in ("undetermined", "wordcand"):
                k = f"{ex}_L{L}_f{f}"; log["judged"] += k in LAB
                if k not in LAB: need[k] = {"example": ex, "layer": L, "f": f, "bucket": b, "domains": doms, "prompt": j}
        top[ex][str(L)] = rows
        if os.path.exists(f"{S}/v2_ex_top10.json"):   # log only: overlap with the earlier mass-ranked top-10 (v2_ex_sel.py)
            log[f"{ex}_overlap_with_mass_top10"] += len(set(rest[:N]) & {r["f"] for r in json.load(open(f"{S}/v2_ex_top10.json"))[ex][str(L)]})
json.dump(top, open(f"{S}/v3_ex_top10.json", "w")); json.dump(need, open(f"{S}/v3_judge_needed.json", "w"), indent=1)
# the judge sample of this round, grouped as pilot_ex_dump_v3.py reads it: {L: {example: {domains, prompt, features: [top-10 records]}}}
sel = {}
for k, v in need.items():
    sel.setdefault(str(v["layer"]), {}).setdefault(v["example"], {"domains": v["domains"], "prompt": v["prompt"], "features": []})
for L, byex in sel.items():
    for ex, e in byex.items():
        fs = {v["f"] for v in need.values() if str(v["layer"]) == L and v["example"] == ex}
        e["features"] = [{"f": x["f"], "track": "shared", **{q: x[q] for q in x if q != "f"}} for x in top[ex][L] if x["f"] in fs]
json.dump({L: sel[L] for L in sorted(sel, key=int)}, open(f"{S}/pilot_ex_sel_v3.json", "w"))
nb = collections.Counter(r["bucket"] for ex in top for rows in top[ex].values() for r in rows)
print("top-10 buckets:", dict(nb), "| already judged:", log["judged"], "| need judge:", len(need),
      dict(collections.Counter(v["example"] for v in need.values())), "| overlap with mass top-10:", {k: v for k, v in log.items() if "overlap" in k}, "of 620 each")
