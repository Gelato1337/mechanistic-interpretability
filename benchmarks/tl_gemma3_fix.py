"""
tl_gemma3_fix.py — patch TransformerLens 2.17's Gemma 3 model to match Hugging Face.

TL's Gemma-3 config (loading_from_pretrained.py) omits two settings that HF's config has:
  1. query_pre_attn_scalar: HF scales attention scores by query_pre_attn_scalar**-0.5
     (27b: 168 -> /12.96); TL defaults attn_scale = sqrt(d_head) (27b: /11.31).
  2. rope_scaling {"rope_type": "linear", "factor": 8} on GLOBAL layers: HF divides positions
     by 8 there; TL has no RoPE scaling.
Diagnosed by comparing TL and HF residuals on the same tokens: embeddings identical, <bos> identical at every
layer (it only attends to itself), all other tokens off by 1.6 % at layer 0 -> ~5 % median deep.
"""
import math
import torch


def fix_tl_gemma3(model, hf_text_cfg):
    out = {}
    qpas = getattr(hf_text_cfg, "query_pre_attn_scalar", None)
    rs = getattr(hf_text_cfg, "rope_scaling", None) or {}
    factor = float(rs.get("factor", 1.0)) if rs.get("rope_type", rs.get("type")) == "linear" else 1.0
    for blk in model.blocks:
        attn = blk.attn
        if qpas is not None:
            attn.attn_scale = math.sqrt(qpas)
        if factor != 1.0 and getattr(attn, "attn_type", "global") == "global":
            cfg = attn.cfg
            pos = torch.arange(cfg.n_ctx, dtype=torch.float32) / factor
            dim = torch.arange(cfg.rotary_dim // 2, dtype=torch.float32)
            freq = cfg.rotary_base ** (dim / (cfg.rotary_dim / 2))
            freq = freq.repeat_interleave(2) if cfg.rotary_adjacent_pairs else freq.repeat(2)
            ang = pos[:, None] / freq[None, :]
            dev, dt = attn.rotary_sin.device, attn.rotary_sin.dtype
            attn.rotary_sin = torch.sin(ang).to(dev, dt)
            attn.rotary_cos = torch.cos(ang).to(dev, dt)
    out["attn_scale"] = math.sqrt(qpas) if qpas else None
    out["global_rope_linear_factor"] = factor
    return out
