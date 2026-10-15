"""
saepipe/encode.py — SAE encoders with one interface. [generic]

    enc.launch(L, x, n_bos) -> handle      enqueue the encode of rows x [n, d_in] (current device = x.device)
    enc.collect(handle)     -> (rows, feats, vals) numpy, rows local to x
TritonJumpReLU: fused kernel, launch() never blocks the host (required for pipelining across GPUs);
                output buffers sized n*cap_per_token + n_bos*bos_cap; overflow -> exact re-encode in collect().
TorchJumpReLU:  dense torch encode + nonzero (host sync per call) — reference / fallback.
act "gemma": val = pre where pre > θ (Gemma Scope, SAELens); "shifted": val = pre − θ (thesis engine).
"""
import torch
from saepipe.kernels import jumprelu as jr


class TritonJumpReLU:
    def __init__(self, saes, act="gemma", fp32_epilogue=True, compact="scan", config="tuned",
                 cap_per_token=1024, bos_cap=163840):
        self.saes, self.cfg = saes, config
        self.kw = dict(shifted=(act == "shifted"), bf16_emu=not fp32_epilogue, compact=compact,
                       tune=(config in (None, "autotune")), config=None if config in (None, "autotune") else config)
        self.cap_tok = {L: cap_per_token for L in saes}
        self.bos_cap, self.overflow_reruns = bos_cap, 0

    def warmup(self):
        """Compile (and autotune if no fixed config) once per GPU, on dummy buffers."""
        seen = set()
        for L, s in self.saes.items():
            d = s["W_enc"].device
            if d in seen:
                continue
            seen.add(d)
            K, N = s["W_enc"].shape
            jr.warmup(d, K, N, s["W_enc"], s["b_enc"], s["threshold"], **self.kw)

    def _launch(self, L, x, capn):
        d = x.device
        bufs = (torch.empty(capn, dtype=torch.int32, device=d), torch.empty(capn, dtype=torch.int32, device=d),
                torch.empty(capn, dtype=torch.float32, device=d))
        cnt = torch.zeros(1, dtype=torch.int32, device=d)
        s = self.saes[L]
        jr.jumprelu_sparse(x, s["W_enc"], s["b_enc"], s["threshold"], *bufs, cnt, **self.kw)
        return bufs, cnt

    def launch(self, L, x, n_bos=0):
        capn = max(x.shape[0] * self.cap_tok[L] + n_bos * self.bos_cap, 1 << 16)
        bufs, cnt = self._launch(L, x, capn)
        return (L, x, bufs, cnt, capn)

    def collect(self, handle):
        L, x, bufs, cnt, capn = handle
        with torch.cuda.device(x.device):
            n = int(cnt.item())
            if n > capn:
                self.overflow_reruns += 1
                self.cap_tok[L] = max(self.cap_tok[L], -(-int(n * 1.25) // x.shape[0]))
                bufs, cnt = self._launch(L, x, int(n * 1.05) + 1024)
                assert int(cnt.item()) == n
            return tuple(b[:n].cpu().numpy() for b in bufs)


class TorchJumpReLU:
    def __init__(self, saes, act="gemma"):
        self.saes, self.shifted, self.overflow_reruns = saes, act == "shifted", 0

    def warmup(self):
        pass

    def launch(self, L, x, n_bos=0):
        s = self.saes[L]
        pre = x.to(s["W_enc"].dtype) @ s["W_enc"] + s["b_enc"]
        lat = torch.relu(pre - s["threshold"]) if self.shifted else torch.relu(pre) * (pre > s["threshold"])
        nz = torch.nonzero(lat)                        # host sync
        return (nz[:, 0].int(), nz[:, 1].int(), lat[nz[:, 0], nz[:, 1]].float())

    def collect(self, handle):
        return tuple(t.cpu().numpy() for t in handle)
