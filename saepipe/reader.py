"""
saepipe/reader.py — read a capture written by scripts/capture.py. [generic]

    from saepipe.reader import Capture
    cap = Capture("runs/example")
    cap.prompts                       # DataFrame: prompt_id, domain, n_tokens, first row
    cap.tokens(0)                     # token strings of prompt 0 (<bos> first)
    c = cap.layer(12)                 # dict of arrays: prompt, pos, feat, val (one entry per firing)
    cap.codes(12, 0)                  # the firings of prompt 0 at layer 12: pos, feat, val
    names, rate = cap.firing_rate(12) # share of each domain's prompts in which each feature fires

A capture directory holds tokens.npz (prompt_ids, domains, n_tokens, token_ids), layer_<L>/part_*.npz (rows, feats,
vals) and run.json. Row r of a layer file is token r of the concatenated prompts; prompt p covers rows
offsets[p] .. offsets[p+1]-1, and its position 0 is <bos>.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .io import load_layer


class Capture:
    def __init__(self, run_dir, tokenizer=None):
        self.dir = Path(run_dir)
        T = np.load(self.dir / "tokens.npz")
        self.prompt_ids = T["prompt_ids"].astype(np.int64)
        self.domains = T["domains"].astype(str)
        self.n_tokens = T["n_tokens"].astype(np.int64)
        self.token_ids = T["token_ids"]
        self.offsets = np.concatenate([[0], np.cumsum(self.n_tokens)])
        self.run = json.loads((self.dir / "run.json").read_text()) if (self.dir / "run.json").exists() else {}
        self.layers = sorted(int(p.name.split("_")[1]) for p in self.dir.glob("layer_*"))
        self._tok = tokenizer

    @property
    def n_prompts(self):
        return len(self.n_tokens)

    @property
    def prompts(self):
        import pandas as pd
        return pd.DataFrame({"prompt_id": self.prompt_ids, "domain": self.domains, "n_tokens": self.n_tokens,
                             "first_row": self.offsets[:-1]})

    @property
    def tokenizer(self):
        if self._tok is None:
            from transformers import AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self.run.get("model_repo", "google/gemma-3-1b-pt"))
        return self._tok

    def tokens(self, p):
        """Token strings of prompt p (index in capture order), word-start marker shown as a space."""
        ids = self.token_ids[self.offsets[p]:self.offsets[p + 1]].tolist()
        return [self.tokenizer.decode([i]) if t.startswith("<0x") else t.replace("▁", " ")
                for i, t in zip(ids, self.tokenizer.convert_ids_to_tokens(ids))]

    @lru_cache(maxsize=2)
    def layer(self, L):
        """Every firing of layer L, sorted by token row: prompt index, position in the prompt, feature, activation."""
        r, f, v = load_layer(self.dir, L)
        r = np.asarray(r, np.int64); p = np.searchsorted(self.offsets, r, side="right") - 1
        return {"prompt": p, "pos": r - self.offsets[p], "feat": np.asarray(f, np.int64), "val": np.asarray(v, np.float32)}

    def codes(self, L, p):
        """The firings of prompt p at layer L: positions, features, activations."""
        c = self.layer(L); a, b = np.searchsorted(c["prompt"], [p, p + 1])
        return c["pos"][a:b], c["feat"][a:b], c["val"][a:b]

    def firing_rate(self, L, n_features=None, skip_bos=True):
        """(domain names, matrix [domains x features]): the share of each domain's prompts in which a feature fires."""
        c = self.layer(L); keep = c["pos"] > 0 if skip_bos else np.ones(len(c["pos"]), bool)
        D = n_features or int(c["feat"].max()) + 1
        names = sorted(set(self.domains)); di = {d: i for i, d in enumerate(names)}; pdom = np.array([di[d] for d in self.domains])
        u = np.unique(c["prompt"][keep] * D + c["feat"][keep])
        M = np.zeros((len(names), D), np.float32); np.add.at(M, (pdom[u // D], u % D), 1)
        return names, M / np.bincount(pdom, minlength=len(names))[:, None]
