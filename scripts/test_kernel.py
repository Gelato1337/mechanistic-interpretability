"""
test_kernel.py — single-GPU correctness + microbenchmark of the fused JumpReLU kernel (saepipe/kernels/jumprelu.py)
against torch, on a REAL Gemma Scope 2 encoder (resid_post 262k l0_small).

Inputs X are synthetic (no model load): Gaussian rows scaled so the mean number of
hits per row is ~TARGET_L0, plus a few rows scaled x20 to mimic <bos>-like rows that
fire tens of thousands of features (as in the thesis captures).

  python scripts/test_kernel.py --model 27b --layers 0,31 --out kernel_test.json      (1 GPU, ~15 min for 27b)
Run it on any new GPU type before trusting the kernel there; every correctness entry should report
only_ref == only_new == 0 against the torch reference of the same precision.
"""
import os, json, time, argparse
import numpy as np
import torch
from saepipe.kernels import jumprelu as jr
from saepipe.sae import GemmaScope2

TARGET_L0 = 60


def keys(rows, feats, N):
    return rows.long() * N + feats.long()


def dense_to_sparse(act):
    nz = act.nonzero()
    return nz[:, 0].int(), nz[:, 1].int(), act[nz[:, 0], nz[:, 1]].float()


def compare(ref_r, ref_f, ref_v, r, f, v, N, pre_ref=None, th=None):
    kr = keys(ref_r, ref_f, N).cpu().numpy()
    kk = keys(r, f, N).cpu().numpy()
    o1, o2 = np.argsort(kr), np.argsort(kk)
    kr, vr = kr[o1], ref_v.cpu().numpy()[o1]
    kk, vk = kk[o2], v.cpu().numpy()[o2]
    common_k, i1, i2 = np.intersect1d(kr, kk, return_indices=True)
    only_ref = np.setdiff1d(kr, kk); only_new = np.setdiff1d(kk, kr)
    out = {"nnz_ref": int(len(kr)), "nnz_new": int(len(kk)),
           "only_ref": int(len(only_ref)), "only_new": int(len(only_new)),
           "jaccard": float(len(common_k) / max(1, len(kr) + len(kk) - len(common_k)))}
    if len(common_k):
        d = np.abs(vr[i1] - vk[i2])
        out["max_abs_diff"] = float(d.max())
        out["max_rel_diff"] = float((d / np.maximum(np.abs(vr[i1]), 1e-6)).max())
    # mismatches should sit at the threshold boundary: report their |pre-th|/th margin
    if pre_ref is not None and (len(only_ref) + len(only_new)):
        mis = np.concatenate([only_ref, only_new])
        rr, ff = mis // N, mis % N
        p = pre_ref[torch.as_tensor(rr, device=pre_ref.device), torch.as_tensor(ff, device=pre_ref.device)].float().cpu().numpy()
        t = th.float().cpu().numpy()[ff]
        out["mismatch_margin_rel_max"] = float((np.abs(p - t) / np.maximum(np.abs(t), 1e-6)).max())
    return out


def run_kernel(x, W, b, th, cap, **kw):
    dev = x.device
    r = torch.empty(cap, dtype=torch.int32, device=dev)
    f = torch.empty(cap, dtype=torch.int32, device=dev)
    v = torch.empty(cap, dtype=torch.float32, device=dev)
    c = torch.zeros(1, dtype=torch.int32, device=dev)
    jr.jumprelu_sparse(x, W, b, th, r, f, v, c, **kw)
    n = int(c.item())
    k = min(n, cap)
    return r[:k], f[:k], v[:k], n


def timeit(fn, reps=10, warm=2):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="27b")
    ap.add_argument("--layers", default="0,31")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--out", default="kernel_test.json")
    a = ap.parse_args()
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    res = {"env": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "hip": getattr(torch.version, "hip", None),
                   "cuda": torch.version.cuda}, "layers": {}}
    torch.manual_seed(0)

    for L in [int(s) for s in a.layers.split(",")]:
        sae = GemmaScope2(a.model).load(L, dev, mode="hf_gpu")
        W, b, th = sae["W_enc"], sae["b_enc"], sae["threshold"]
        K, N = W.shape
        R = res["layers"][L] = {"K": K, "N": N, "threshold_min": float(th.min()),
                                "n_threshold_le0": int((th <= 0).sum())}

        # scale Gaussian X so mean hits/row ~ TARGET_L0 (bisection on scale)
        z = torch.randn(512, K, device=dev, dtype=torch.float32)
        lo, hi = 0.01, 1000.0
        for _ in range(30):
            s = (lo * hi) ** 0.5
            pre = (z * s).bfloat16() @ W + b
            hits = ((pre > th) & (pre > 0)).sum(1).float().mean()
            lo, hi = (s, hi) if hits < TARGET_L0 else (lo, s)
        scale = (lo * hi) ** 0.5
        R["x_scale"] = scale

        def make_x(M, n_bos=0):
            x = torch.randn(M, K, device=dev) * scale
            if n_bos:
                x[torch.randperm(M, device=dev)[:n_bos]] *= 20
            return x.bfloat16().contiguous()

        # ---- autotune per mode on dummy buffers ----
        t0 = time.perf_counter()
        for shifted in (0, 1):
            for emu in (0, 1):
                for comp in ("scan", "atomic"):
                    jr.warmup(dev, K, N, W, b, th, shifted=shifted, bf16_emu=emu, compact=comp)
        R["autotune_s"] = time.perf_counter() - t0
        R["autotune_choice"] = jr.best_config()

        # ---- correctness ----
        x = make_x(1024, n_bos=4)
        cap = 1 << 24
        corr = {}
        pre_bf = x @ W + b
        for shifted in (0, 1):
            ref_bf = jr.torch_reference(x, W, b, th, shifted=bool(shifted))
            ref_32 = jr.torch_reference(x, W, b, th, shifted=bool(shifted), fp32=True)
            rb = dense_to_sparse(ref_bf); r32 = dense_to_sparse(ref_32)
            for emu in (1, 0):
                for comp in ("scan", "atomic"):
                    r, f, v, n = run_kernel(x, W, b, th, cap, shifted=shifted, bf16_emu=emu, compact=comp)
                    ref = rb if emu else r32
                    name = f"shifted{shifted}_emu{emu}_{comp}_vs_torch_{'bf16' if emu else 'fp32'}"
                    corr[name] = compare(*ref, r, f, v, N, pre_ref=pre_bf, th=th)
                    print(L, name, corr[name], flush=True)
            corr[f"shifted{shifted}_torch_bf16_vs_fp32"] = compare(*r32, *rb, N, pre_ref=pre_bf, th=th)
        R["correctness"] = corr
        del pre_bf

        # ---- overflow: cap below true count -> counter reports true count, no crash ----
        r, f, v, n = run_kernel(x, W, b, th, 1000, shifted=0, bf16_emu=1)
        R["overflow"] = {"cap": 1000, "counter": n, "true_nnz": corr["shifted0_emu1_scan_vs_torch_bf16"]["nnz_ref"],
                         "kept_valid": bool((f < N).all() and (f >= 0).all())}

        # ---- speed ----
        speed = {}
        for M in (256, 1024, 4096):
            x = make_x(M, n_bos=max(1, M // 256))

            def gemm_only():
                return x @ W

            def torch_batched():       # same math, all tokens in one GEMM
                lat = torch.relu(x @ W + b - th)
                nz = torch.nonzero(lat)
                return nz.cpu(), lat[nz[:, 0], nz[:, 1]].cpu()

            capM = M * 4096
            bufs = [torch.empty(capM, dtype=t, device=dev) for t in (torch.int32, torch.int32, torch.float32)]
            cnt = torch.zeros(1, dtype=torch.int32, device=dev)

            def triton_sparse(comp="scan", emu=1):
                cnt.zero_()
                jr.jumprelu_sparse(x, W, b, th, *bufs, cnt, shifted=1, bf16_emu=emu, compact=comp)
                n = int(cnt.item())
                return bufs[0][:n].cpu(), bufs[1][:n].cpu(), bufs[2][:n].cpu()

            flops = 2 * M * K * N
            row = {"gemm_only_s": timeit(gemm_only, a.reps),
                   "torch_batched_s": timeit(torch_batched, a.reps),
                   "triton_scan_emu1_s": timeit(lambda: triton_sparse("scan", 1), a.reps),
                   "triton_scan_emu0_s": timeit(lambda: triton_sparse("scan", 0), a.reps),
                   "triton_atomic_emu1_s": timeit(lambda: triton_sparse("atomic", 1), a.reps)}
            row["gemm_only_tflops"] = flops / row["gemm_only_s"] / 1e12
            row["triton_scan_emu1_tflops"] = flops / row["triton_scan_emu1_s"] / 1e12
            row["nnz"] = int(cnt.item())
            speed[M] = row
            print(L, M, row, flush=True)
            del x, bufs
            torch.cuda.empty_cache()
        R["speed"] = speed
        del W, b, th, sae
        torch.cuda.empty_cache()

    json.dump(res, open(a.out, "w"), indent=2, default=str)
    print("wrote", a.out)
    print("wrote", out)


if __name__ == "__main__":
    main()
