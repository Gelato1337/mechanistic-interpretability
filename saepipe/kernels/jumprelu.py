"""
saepipe/kernels/jumprelu.py — fused JumpReLU SAE encode with sparse output. [generic: Triton, ROCm + CUDA]
Launch parameters per vendor live in configs.py.

One kernel does:  pre = X @ W_enc + b_enc  ->  hit = pre > threshold  ->  write only the
hits as (row, feat, val) into preallocated buffers. The dense [M, d_sae] pre-activation
matrix is never written to memory and there is no host sync (no torch.nonzero).

Slot allocation: each program counts its hits, reserves a contiguous range with ONE
atomic_add on a global counter, and places hits inside it with a 2-D prefix sum
(COMPACT=0, "scan"). COMPACT=1 ("atomic") uses one atomic per hit instead — slower
under contention but needs no scan; kept as a fallback / comparison.
Output order is nondeterministic across programs; callers sort if they need order.
If the counter ends above `cap`, hits beyond cap were dropped: re-run with cap >= counter.

Numerics (BF16_EMU):
  1 — mimic the torch bf16 path (thesis engine): acc fp32 -> round bf16, + b_enc, round bf16,
      compare with bf16 threshold; SHIFTED value pre-th rounded to bf16.
      Differs from torch only through GEMM accumulation order.
  0 — fp32 epilogue: pre = acc + b_enc in fp32 (more accurate than torch bf16).
Activation definition (SHIFTED):
  0 — Gemma Scope / SAELens JumpReLU:  val = pre       where pre > th  (and pre > 0)
  1 — thesis engine:                   val = pre - th  where pre > th
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _jumprelu_sparse_kernel(
    X, W, B, TH,
    out_rows, out_feats, out_vals, count_ptr,
    M, N, K, cap, row_offset,
    stride_xm, stride_xk, stride_wk, stride_wn,
    SHIFTED: tl.constexpr, BF16_EMU: tl.constexpr, COMPACT: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # grouped program order: GROUP_M row-tiles share each W column-tile while it is in L2
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    x_ptrs = X + offs_m[:, None].to(tl.int64) * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = W + offs_k[:, None].to(tl.int64) * stride_wk + offs_n[None, :].to(tl.int64) * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_rem), other=0.0)
        w = tl.load(w_ptrs, mask=(offs_k[:, None] < k_rem) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(x, w, acc)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    nmask = offs_n < N
    b = tl.load(B + offs_n, mask=nmask, other=0.0).to(tl.float32)
    th = tl.load(TH + offs_n, mask=nmask, other=float("inf")).to(tl.float32)
    if BF16_EMU:
        pre = (acc.to(tl.bfloat16).to(tl.float32) + b[None, :]).to(tl.bfloat16).to(tl.float32)
    else:
        pre = acc + b[None, :]
    valid = (offs_m[:, None] < M) & nmask[None, :]
    if SHIFTED:
        hit = (pre > th[None, :]) & valid
        val = pre - th[None, :]
        if BF16_EMU:
            val = val.to(tl.bfloat16).to(tl.float32)
    else:
        hit = (pre > th[None, :]) & (pre > 0.0) & valid
        val = pre

    rows = offs_m[:, None] + row_offset + tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    feats = offs_n[None, :] + tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    hit_i = hit.to(tl.int32)

    if COMPACT == 0:
        row_cnt = tl.sum(hit_i, axis=1)
        total = tl.sum(row_cnt, axis=0)
        if total > 0:
            base = tl.atomic_add(count_ptr, total)
            row_excl = tl.cumsum(row_cnt, axis=0) - row_cnt
            within = tl.cumsum(hit_i, axis=1) - hit_i
            slot = base + row_excl[:, None] + within
            ok = hit & (slot < cap)
            tl.store(out_rows + slot, rows, mask=ok)
            tl.store(out_feats + slot, feats, mask=ok)
            tl.store(out_vals + slot, val, mask=ok)
    else:
        ones = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32) + 1
        slot = tl.atomic_add(count_ptr + tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32), ones, mask=hit)
        ok = hit & (slot < cap)
        tl.store(out_rows + slot, rows, mask=ok)
        tl.store(out_feats + slot, feats, mask=ok)
        tl.store(out_vals + slot, val, mask=ok)


from saepipe.backend import vendor as _vendor
from saepipe.kernels import configs as _configs

_AUTOTUNERS = {}


def _autotuner(v):
    if v not in _AUTOTUNERS:
        _AUTOTUNERS[v] = triton.autotune(configs=_configs.AUTOTUNE[v], key=["N", "K", "SHIFTED", "BF16_EMU", "COMPACT"],
                                         reset_to_zero=["count_ptr"])(_jumprelu_sparse_kernel)
    return _AUTOTUNERS[v]


def resolve_config(config):
    """config: None/"autotune" -> autotune; "tuned" -> configs.TUNED[vendor] (autotune if none); dict -> fixed."""
    v = _vendor()
    if config in (None, "autotune"):
        return None
    if config == "tuned":
        t = _configs.TUNED.get(v)
        return None if t is None else _configs.sanitize(v, t)
    return _configs.sanitize(v, dict(_configs.FIXED_DEFAULT[v], **config))


def jumprelu_sparse(x, W, b, th, out_rows, out_feats, out_vals, counter, row_offset=0,
                    shifted=False, bf16_emu=True, compact="scan", tune=True, config=None):
    """Enqueue the fused encode. `counter` is a 1-element int32 view (zeroed by caller).
    Must be called with x's device as the current device (torch.cuda.device(x.device)).
    config: None/"autotune" | "tuned" | dict of launch parameters (missing keys from FIXED_DEFAULT)."""
    assert x.is_contiguous() and W.dim() == 2
    M, K = x.shape
    N = W.shape[1]
    cap = out_rows.numel()
    args = (x, W, b, th, out_rows, out_feats, out_vals, counter,
            M, N, K, cap, row_offset,
            x.stride(0), x.stride(1), W.stride(0), W.stride(1))
    flags = dict(SHIFTED=int(shifted), BF16_EMU=int(bf16_emu), COMPACT=0 if compact == "scan" else 1)
    c = None if (tune and config is None) else resolve_config(config if config is not None else {})
    if c is None:
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
        _autotuner(_vendor())[grid](*args, **flags)
    else:
        grid = (triton.cdiv(M, c["BLOCK_M"]) * triton.cdiv(N, c["BLOCK_N"]),)
        _jumprelu_sparse_kernel[grid](*args, **flags, **c)


def best_config():
    """Autotuner's choice per key (after a warmup call)."""
    a = _AUTOTUNERS.get(_vendor())
    return {str(k): str(v) for k, v in getattr(a, "cache", {}).items()} if a else {}


def warmup(device, K, N, W=None, b=None, th=None, shifted=False, bf16_emu=True, compact="scan", M=2048,
           tune=True, config=None):
    """Populate the autotune cache with dummy buffers so real calls never benchmark
    (autotuning reruns the kernel and would corrupt a live counter)."""
    with torch.cuda.device(device):
        x = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        if W is None:
            W = torch.randn(K, N, device=device, dtype=torch.bfloat16) * K ** -0.5
            b = torch.zeros(N, device=device, dtype=torch.bfloat16)
            th = torch.full((N,), 3.0, device=device, dtype=torch.bfloat16)
        cap = 1 << 22
        r = torch.empty(cap, dtype=torch.int32, device=device)
        f = torch.empty(cap, dtype=torch.int32, device=device)
        v = torch.empty(cap, dtype=torch.float32, device=device)
        c = torch.zeros(1, dtype=torch.int32, device=device)
        jumprelu_sparse(x, W, b, th, r, f, v, c, shifted=shifted, bf16_emu=bf16_emu, compact=compact,
                        tune=tune, config=config)
        torch.cuda.synchronize(device)


def torch_reference(x, W, b, th, shifted=False, fp32=False):
    """Dense torch encode (thesis-style bf16 by default). Returns dense activations."""
    if fp32:
        pre = x.float() @ W.float() + b.float()
        th = th.float()
    else:
        pre = x @ W + b
    if shifted:
        return torch.relu(pre - th)
    return torch.relu(pre) * (pre > th)
