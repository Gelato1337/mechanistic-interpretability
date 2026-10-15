#!/usr/bin/env python3
"""
aggregate.py — reduce a capture (scripts/capture.py output) to per-domain feature statistics, one file per layer.
Every analysis of the thesis that does not need single tokens starts from these files.

    python scripts/aggregate.py --capture runs/27b_all --prompts prompts.csv --out stats/262k/27b

Definitions (per layer, domain d, feature f; position 0 = <bos> is never counted; a prompt has T tokens and
nt = T - 1 positions after <bos>):
  NFD[d, f]   prompts of d in which f fires on any position 1..nt
  NF1[d, f]   prompts of d in which f fires on position 1 (the first real token)
  MASS[d, f]  mean over the prompts of d of  (sum of f's activation over positions 2..nt) / nt
              ("mass": the feature's mean write into the residual stream per token; the decoder rows are unit norm)
  NPR[d]      prompts of d
  nf[f]       sum_d NFD[d, f];  rate[f] = nf / sum_d NPR  (the corpus firing rate; always-on = rate >= .6)
  NFD_h, MASS_h, NPR_h   the same over the odd and the even prompts of each domain (split-half checks)
Half h of a prompt = its index within the domain (prompt_id order) mod 2.

--drop-duplicates removes the abstract_philosophy prompts whose text also occurs in abstract_religion (the 14
duplicates of the thesis dataset; the thesis analyses use 13,986 prompts).

Output: <out>/L<layer>.npz with names, NFD, NF1, MASS, NPR, nf, rate, NFD_h, MASS_h, NPR_h, n_layers.
"""
import argparse, glob, json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True, help="capture directory (tokens.npz + layer_<L>/)")
    ap.add_argument("--prompts", nargs="+", default=None, help="prompt CSV(s); needed for --drop-duplicates")
    ap.add_argument("--drop-duplicates", type=int, default=1)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--n-features", type=int, default=262144)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cap, out = Path(a.capture), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    T = np.load(cap / "tokens.npz"); ntok = T["n_tokens"].astype(np.int64); dom = T["domains"].astype(str); pid = T["prompt_ids"]
    off = np.concatenate([[0], np.cumsum(ntok)]); P = len(ntok); D = a.n_features

    keep = np.ones(P, bool)
    if a.drop_duplicates:
        if not a.prompts:
            raise SystemExit("--drop-duplicates needs --prompts")
        df = pd.concat([pd.read_csv(p) for p in a.prompts], ignore_index=True)
        rel = {t.strip() for t in df[df.micro_domain == "abstract_religion"].text}
        dup = set(df[(df.micro_domain == "abstract_philosophy") & df.text.str.strip().isin(rel)].prompt_id.astype(int))
        keep = np.array([int(x) not in dup for x in pid]); print(f"dropped {int((~keep).sum())} duplicated philosophy prompts", flush=True)

    names = sorted(set(dom[keep])); di = {d: i for i, d in enumerate(names)}; n = len(names)
    pdom = np.array([di.get(d, -1) for d in dom]); half = np.zeros(P, int)
    for d in names:
        idx = np.where((dom == d) & keep)[0]; idx = idx[np.argsort(pid[idx], kind="stable")]; half[idx] = np.arange(len(idx)) % 2
    NPR = np.bincount(pdom[keep], minlength=n); NPR_h = np.stack([np.bincount(pdom[keep & (half == h)], minlength=n) for h in (0, 1)])
    run = json.loads((cap / "run.json").read_text()) if (cap / "run.json").exists() else {}
    layers = sorted(int(Path(p).name.split("_")[1]) for p in glob.glob(str(cap / "layer_*"))) if a.layers == "all" else [int(x) for x in a.layers.split(",")]

    for L in layers:
        NFD = np.zeros((n, D), np.int32); NF1 = np.zeros((n, D), np.int32); MASS = np.zeros((n, D), np.float64)
        NFD_h = np.zeros((2, n, D), np.int32); MASS_h = np.zeros((2, n, D), np.float64)
        for part in sorted((cap / f"layer_{L}").glob("part_*.npz")):
            z = np.load(part); r = z["rows"].astype(np.int64); f = z["feats"].astype(np.int64); v = z["vals"].astype(np.float64)
            p = np.searchsorted(off, r, side="right") - 1; pos = r - off[p]
            m = keep[p] & (pos >= 1); r, f, v, p, pos = r[m], f[m], v[m], p[m], pos[m]
            nt = ntok[p] - 1
            # fired in the prompt (positions 1..nt): unique (prompt, feature) pairs
            k = np.unique(p * D + f); kp, kf = k // D, k % D
            np.add.at(NFD, (pdom[kp], kf), 1); np.add.at(NFD_h, (half[kp], pdom[kp], kf), 1)
            # first real token
            m1 = pos == 1; np.add.at(NF1, (pdom[p[m1]], f[m1]), 1)
            # mass: positions 2..nt, divided by nt, summed per prompt then averaged per domain below
            m2 = (pos >= 2) & (nt >= 2); w = v[m2] / nt[m2]
            np.add.at(MASS, (pdom[p[m2]], f[m2]), w); np.add.at(MASS_h, (half[p[m2]], pdom[p[m2]], f[m2]), w)
        nf = NFD.sum(0); rate = (nf / NPR.sum()).astype(np.float32)
        np.savez_compressed(out / f"L{L}.npz", names=np.array(names), NFD=NFD, NF1=NF1, MASS=(MASS / NPR[:, None]).astype(np.float32), NPR=NPR,
                            nf=nf, rate=rate, NFD_h=NFD_h, MASS_h=(MASS_h / np.maximum(NPR_h, 1)[:, :, None]).astype(np.float32), NPR_h=NPR_h,
                            n_layers=run.get("n_layers", -1))
        print(f"L{L}: features fired {int((nf > 0).sum()):,}, always-on (rate >= .6) {int((rate >= .6).sum())}", flush=True)


if __name__ == "__main__":
    main()
