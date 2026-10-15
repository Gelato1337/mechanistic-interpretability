#!/usr/bin/env python3
"""build_topic_dataset3.py — the topic-feature dataset for all five sizes under the final rule (one content home with 25 to 300 home prompts
and corpus rate <= .05; at least 4 fired tokens per home prompt and coverage >= 3 % of the home prompts' tokens; fires in >= 2 prompts of
>= 5 other domains): joins work/topic_rule_sizes.json (topic_sizes_eval.py) with work/topic_passages3/ (topic_passages3.py
scripts/capture.py captures). Output work/topic_features_dataset.jsonl, topic_features_index.csv, topic_features_README.md."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../lib"))  # thesis/analysis/lib
from paths import WORK
import json, csv, collections
d = json.load(open(f"{WORK}/topic_rule_sizes.json")); SL = {"270m": "270M", "1b": "1B", "4b": "4B", "12b": "12B", "27b": "27B"}
out = open(f"{WORK}/topic_features_dataset.jsonl", "w"); idx = []; per = collections.Counter(); miss = 0; cache = {}
for s, feats in d["features"].items():
    for x in feats:
        L = x["layer"]; key = (s, L)
        if key not in cache: cache[key] = {r["f"]: r for r in json.load(open(f"{WORK}/topic_passages3_{s}_L{L}.json"))["features"]}
        p = cache[key].get(x["f"]); miss += p is None
        rec = dict(model=SL[s], layer=L, n_layers=d["sizes"][s]["layers"][str(L)]["n_layers"], feature=x["f"], home=x["home"], home_prompts=x["home_n"], prompts_total=x["tot"], fired_tokens_per_home_prompt=round(x["mean_nf"], 2), coverage_of_home_prompts=round(x["mean_cov"], 4),
                   prompts_per_domain=x["counts"], prompts_per_unit=x["any"], domains_reached=x["reached"], passages=(p["passages"] if p else {}), n_fired_per_prompt=(p["n_fired"] if p else {}))
        out.write(json.dumps(rec, ensure_ascii=False) + "\n"); per[SL[s]] += 1
        idx.append(dict(model=SL[s], layer=L, feature=x["f"], home=x["home"], home_prompts=x["home_n"], prompts_total=x["tot"], fired_tokens_per_home_prompt=round(x["mean_nf"], 2), coverage=round(x["mean_cov"], 4), domains_reached=len(x["reached"]), reached=" ".join(x["reached"])))
out.close()
with open(f"{WORK}/topic_features_index.csv", "w", newline="") as f: w = csv.DictWriter(f, fieldnames=list(idx[0].keys())); w.writeheader(); w.writerows(idx)
open(f"{WORK}/topic_features_README.md", "w").write(f"""# Topic-feature dataset (Gemma 3, five sizes; Gemma Scope 2 262k resid_post SAEs)

One record per topic feature at eight layers of each size, the layers at the relative depths of layers 31, 35, 39, 43, 47, 51, 55 and 59 of the 62-layer 27B model. Counts: {dict(per)} ({sum(per.values())} features).

Rule: the feature's home is the content domain holding at least half of the prompts it fires in; it fires in 25 to 300 of the home's 500 prompts and in at most 5 % of all 14,000 prompts; it fires on at least 4 tokens per home prompt on average and on at least 3 % of the home prompts' tokens; it fires in at least 2 prompts of each of at least 5 other domains. The five FLORES languages count as one domain (par_lang) and the four Rosetta languages as one (par_code). Captures: October 2026 per-token captures of the 28 thesis domains with the optimised engine (sae-optimization), start token skipped.

Files: topic_features_dataset.jsonl (full records), topic_features_index.csv (one line per feature). Fields: model, layer, n_layers, feature (SAE index), home, home_prompts, prompts_total, fired_tokens_per_home_prompt, coverage_of_home_prompts, prompts_per_domain (28 domains), prompts_per_unit (21 units, home excluded), domains_reached, passages (per domain: up to 3 home passages and 2 passages per other domain reached, each with the token list and the activation on every fired token, cut at 320 tokens), n_fired_per_prompt (per domain, the number of fired tokens in each prompt the feature fires in).
""")
print("features", sum(per.values()), dict(per), "without passages", miss)
