#!/usr/bin/env python3
"""topic_super.py — the candidate list for topic_passages3.py: per size and layer, the features of hunt_stats_sizes.py that pass a
looser version of the topic-feature rule (so that passages exist for every feature the final rule in topic_sizes_eval.py can keep):
one content home with 25 to 300 home prompts and rate <= .05 (hunt_stats_sizes.py), at least 3 fired tokens and 2 % coverage per
home prompt, fires in >= 2 prompts of >= 5 other domains. Output work/topic_super_{size}_L{L}.json {"size", "layer", "features": [{f, home}]}."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../lib"))  # thesis/analysis/lib
import json
from paths import WORK
SIZES = {"270m": [9, 10, 11, 12, 13, 14, 15, 16], "1b": [13, 14, 16, 18, 19, 21, 23, 24], "4b": [17, 19, 21, 23, 25, 28, 30, 32],
         "12b": [24, 27, 30, 33, 36, 39, 42, 45], "27b": [31, 35, 39, 43, 47, 51, 55, 59]}
HMIN, HMAX, NF, COV, UNITS_MIN = 25, 300, 3, .02, 5
for s, Ls in SIZES.items():
    for L in Ls:
        F = json.load(open(f"{WORK}/hunt_stats3_{s}_L{L}.json"))["features"]
        keep = [dict(f=x["f"], home=x["home"]) for x in F if HMIN <= x["home_n"] <= HMAX and x["mean_nf"] >= NF and x["mean_cov"] >= COV
                and sum(n >= 2 for n in x["any"].values()) >= UNITS_MIN]
        json.dump({"size": s, "layer": L, "features": keep}, open(f"{WORK}/topic_super_{s}_L{L}.json", "w")); print(s, L, len(keep), flush=True)
