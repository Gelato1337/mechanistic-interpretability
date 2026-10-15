#!/usr/bin/env python3
"""tokdump.py (CPU, about 60 GB RAM for 27B) — every feature on every token of the two worked examples, all layers of Gemma 3 27B,
read from the 27B capture (scripts/capture.py; paths.capture('27b')).
    opera = par_lang_{english,finnish,french,german,swedish}, prompt index 21 within each domain (prompt_id order)
    prime = par_code_{python,javascript,cpp,c},                prompt index 28
    control: the next prompt (22 / 29) of the same domains
Output work/tokdump_all.json
    {"examples": {ex: {"domains": [...], "prompt": j}},
     "layers": {L: {"texts": {domain or domain_ctrl: [{"t": token string, "f": [[feature, activation], ...]}, ...]},
                    "n_prompts": prompts in the capture,
                    "rarity": {feature: [prompts it fires in, domains it fires in]}}}}
and work/tokdump_v2.json (the example texts only). Position 0 is <bos>, which carries no firings (it is not encoded); downstream
scripts exclude it."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../lib"))  # thesis/analysis/lib
import json
import numpy as np
from paths import WORK, capture
from shards import _index, layers, load_layer, token_strings

EX = {"opera": ([f"par_lang_{l}" for l in ("english", "finnish", "french", "german", "swedish")], 21),
      "prime": ([f"par_code_{c}" for c in ("python", "javascript", "cpp", "c")], 28)}
CAP = capture("27b"); D = 262144
from pathlib import Path
T, ntok, off = _index(Path(CAP)); dom = T["domains"].astype(str); pid = T["prompt_ids"].astype(np.int64); strings = token_strings(CAP)
names = sorted(set(dom)); pdom = np.array([names.index(d) for d in dom])
def row_of(d, j):
    idx = np.where(dom == d)[0]; return int(idx[np.argsort(pid[idx], kind="stable")][j])
want = {(d + tag): row_of(d, j + dj) for ex, (dl, j) in EX.items() for d in dl for tag, dj in (("", 0), ("_ctrl", 1))}
out = {"examples": {k: {"domains": v[0], "prompt": v[1]} for k, v in EX.items()}, "layers": {}}
v2 = {"examples": out["examples"], "layers": {}}
for L in layers(CAP):
    r, f, v = load_layer(CAP, L); b = np.searchsorted(r, off); rec = {"texts": {}}; used = set()
    for key, k in want.items():
        pos = r[b[k]:b[k + 1]] - off[k]; ff = f[b[k]:b[k + 1]]; vv = v[b[k]:b[k + 1]]
        toks = [{"t": t, "f": []} for t in strings[k]]
        for q in np.argsort(-vv, kind="stable"): toks[int(pos[q])]["f"].append([int(ff[q]), round(float(vv[q]), 2)])
        rec["texts"][key] = toks; used.update(ff.tolist())
    p = np.searchsorted(off, r, side="right") - 1; u = np.unique(p * D + f); up, uf = u // D, u % D
    nprom = np.bincount(uf, minlength=D); ud = np.unique(pdom[up] * D + uf) % D; ndom = np.bincount(ud, minlength=D)
    rec["n_prompts"] = int(len(ntok)); rec["rarity"] = {int(x): [int(nprom[x]), int(ndom[x])] for x in sorted(used)}
    out["layers"][str(L)] = rec; v2["layers"][str(L)] = {"texts": {k: t for k, t in rec["texts"].items() if not k.endswith("_ctrl")}}
    print(L, len(used), "features", flush=True)
json.dump(out, open(f"{WORK}/tokdump_all.json", "w")); json.dump(v2, open(f"{WORK}/tokdump_v2.json", "w")); print("TOKDUMP_DONE")
