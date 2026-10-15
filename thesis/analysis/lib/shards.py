"""shards.py — per-token access to a capture written by scripts/capture.py.

load_domains(capture_dir, layer) returns the layout the per-token analyses use:
    {domain: {"token_strings": [[str, ...], ...],          # one list per prompt, <bos> at position 0
              "activations":   [(pos, fid, act), ...],     # one triple of arrays per prompt; pos indexes token_strings
              "prompt_ids":    [int, ...]}}
Prompts of a domain are in prompt_id order, so index j of a domain is the same prompt in every layer and every size.
Token strings are the tokenizer's pieces with the word-start marker turned into a space ("▁the" -> " the"); byte
pieces are decoded singly. <bos> is not encoded (capture.py --skip-bos 1), so every pos is >= 1."""
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

_BYTE = re.compile(r"<0x[0-9A-Fa-f]{2}>")


@lru_cache(maxsize=None)
def tokenizer(repo="google/gemma-3-1b-pt"):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(repo)     # the five Gemma 3 sizes share one tokenizer


def _index(cap):
    T = np.load(cap / "tokens.npz")
    ntok = T["n_tokens"].astype(np.int64)
    return T, ntok, np.concatenate([[0], np.cumsum(ntok)])


@lru_cache(maxsize=4)
def _strings(capture_dir):
    cap = Path(capture_dir); T, ntok, off = _index(cap)
    run = json.loads((cap / "run.json").read_text()) if (cap / "run.json").exists() else {}
    tok = tokenizer(run.get("model_repo", "google/gemma-3-1b-pt"))
    u, inv = np.unique(T["token_ids"], return_inverse=True)
    pieces = tok.convert_ids_to_tokens(u.tolist())
    # byte-fallback pieces ("<0xC2>") are decoded one by one, as the thesis captures stored them
    us = [tok.decode([i]) if _BYTE.fullmatch(t) else t.replace("▁", " ") for i, t in zip(u.tolist(), pieces)]
    return [[us[i] for i in inv[off[k]:off[k + 1]]] for k in range(len(ntok))]


def token_strings(capture_dir):
    """Token strings of every prompt of a capture, in capture order (cached)."""
    return _strings(str(Path(capture_dir).resolve()))


def layers(capture_dir):
    """Layer indices present in a capture."""
    return sorted(int(p.name.split("_")[1]) for p in Path(capture_dir).glob("layer_*"))


def load_layer(capture_dir, layer):
    """All firings of one layer sorted by global token row: rows, feats, vals."""
    parts = sorted((Path(capture_dir) / f"layer_{layer}").glob("part_*.npz"))
    if not parts:
        raise FileNotFoundError(f"no shards for layer {layer} in {capture_dir}")
    zs = [np.load(p) for p in parts]
    r = np.concatenate([z["rows"] for z in zs]).astype(np.int64)
    f = np.concatenate([z["feats"] for z in zs]).astype(np.int64)
    v = np.concatenate([z["vals"] for z in zs]).astype(np.float32)
    o = np.argsort(r, kind="stable")
    return r[o], f[o], v[o]


def load_domains(capture_dir, layer, drop=()):
    """The per-domain per-prompt layout above. `drop`: domains to leave out."""
    cap = Path(capture_dir); T, ntok, off = _index(cap)
    dom = T["domains"].astype(str); pid = T["prompt_ids"].astype(np.int64)
    strings = token_strings(cap)
    r, f, v = load_layer(cap, layer); b = np.searchsorted(r, off)
    out = {}
    for d in sorted(set(dom) - set(drop)):
        idx = np.where(dom == d)[0]; idx = idx[np.argsort(pid[idx], kind="stable")]
        out[d] = {"token_strings": [strings[k] for k in idx], "prompt_ids": pid[idx].tolist(),
                  "activations": [(r[b[k]:b[k + 1]] - off[k], f[b[k]:b[k + 1]], v[b[k]:b[k + 1]]) for k in idx]}
    return out
