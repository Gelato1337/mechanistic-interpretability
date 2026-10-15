"""
saepipe/kernels/configs.py — Triton launch configurations per GPU vendor.

The kernel source is vendor-neutral; launch parameters are not:
  * waves_per_eu, matrix_instr_nonkdim, kpack exist only in Triton's AMD backend;
  * good tile shapes / num_stages differ between MI250X and NVIDIA GPUs.
TUNED[vendor]: a fixed config validated for exact output and speed (None = not tuned yet -> autotune).
AUTOTUNE[vendor]: candidates for triton.autotune. Every new config must pass check_configs.py (exact
support vs torch) before being added to TUNED.
KNOWN_BAD[vendor]: parameter combinations observed to miscompile (garbage output).
"""
import triton

TUNED = {
    # AMD MI250X (LUMI), Gemma Scope 2 27b 262k encoders (K=5376, N=262144); validated with scripts/test_kernel.py and a config sweep
    # 23.3 ms at M=640 vs 41.5 ms for the batched torch encode; exact at M=640 and 2560.
    "rocm": dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, GROUP_M=8, num_warps=8, num_stages=2,
                 waves_per_eu=2, matrix_instr_nonkdim=16, kpack=2),
    "cuda": None,      # not tuned/validated on NVIDIA yet
}


def _c(bm, bn, bk, warps, stages, group=8, **extra):
    return triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": group, **extra},
                         num_warps=warps, num_stages=stages)


AUTOTUNE = {
    "rocm": [_c(64, 128, 64, 4, 2), _c(128, 128, 64, 8, 2), _c(64, 256, 64, 8, 2), _c(128, 256, 32, 8, 2),
             _c(32, 128, 64, 4, 2), _c(64, 128, 32, 4, 2), _c(128, 128, 32, 4, 2), _c(64, 64, 64, 4, 2)],
    # standard Ampere/Hopper GEMM shapes (Triton matmul tutorial family); unvalidated
    "cuda": [_c(128, 256, 64, 8, 3), _c(64, 256, 32, 4, 4), _c(128, 128, 32, 4, 4), _c(128, 64, 32, 4, 4),
             _c(64, 128, 32, 4, 4), _c(128, 128, 64, 8, 3), _c(64, 64, 64, 4, 3)],
}

FIXED_DEFAULT = {
    "rocm": dict(BLOCK_M=64, BLOCK_N=128, BLOCK_K=64, GROUP_M=8, num_warps=4, num_stages=2),
    "cuda": dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, GROUP_M=8, num_warps=4, num_stages=4),
}

KNOWN_BAD = {
    # all 8 configs with these values gave Jaccard ~0.01 / millions of duplicate slots
    "rocm": [dict(BLOCK_M=128, BLOCK_N=256, num_warps=8, num_stages=1)],
    "cuda": [],
}

AMD_ONLY_KEYS = ("waves_per_eu", "matrix_instr_nonkdim", "kpack")


def is_known_bad(vendor, cfg):
    return any(all(cfg.get(k) == v for k, v in bad.items()) for bad in KNOWN_BAD.get(vendor, []))


def sanitize(vendor, cfg):
    """Drop AMD-only launch keys on CUDA; refuse known-bad combinations."""
    cfg = dict(cfg)
    if vendor != "rocm":
        for k in AMD_ONLY_KEYS:
            cfg.pop(k, None)
    if is_known_bad(vendor, cfg):
        raise ValueError(f"config {cfg} is on the known-miscompile list for {vendor}")
    return cfg

# Triton CPU interpreter (TRITON_INTERPRET=1, offline logic tests): behave like CUDA
TUNED["cpu"] = None
AUTOTUNE["cpu"] = AUTOTUNE["cuda"]
FIXED_DEFAULT["cpu"] = FIXED_DEFAULT["cuda"]
KNOWN_BAD["cpu"] = []
