"""
saepipe/pipeline.py — forward-only GPipe scheduler across GPUs with per-layer SAE encodes. [generic for HF
decoders; hook point = decoder-layer output (resid_post) only]

Micro-batch m enters stage s at step t = m + s, so every stage (GPU) works on a different micro-batch at
every step. Requirements: the host loop must never block on the GPU -> use an async encoder (TritonJumpReLU);
TorchJumpReLU works but serialises the GPUs (host sync per layer).

    pipe = Pipeline(tm, layers, encoder, skip_bos=False)
    for wave in waves:                         # list of (BatchEncoding, [row ids per prompt])
        pipe.run_wave(wave, emit)              # emit(L, global_rows, feats, vals) after the wave's readback
    pipe.close()
"""
import numpy as np
import torch
from saepipe.model import Layer0Capture, build_stages, layer_device, to_device


class Pipeline:
    def __init__(self, tm, layers, encoder, skip_bos=False):
        self.tm, self.enc, self.skip_bos = tm, encoder, skip_bos
        self.layers, self.lset, self.last = list(layers), set(layers), max(layers)
        self.stages = build_stages(tm, self.last)
        self.stage_devs = [d for d, _ in self.stages]
        self.enc_devs = {layer_device(tm, L) for L in layers}
        self.cap = Layer0Capture(tm)

    def prepare(self, enc, rows_per_prompt):
        """Tokenized micro-batch -> layer-0 inputs on every stage GPU, flat valid-token index, row map."""
        am = enc["attention_mask"].clone()
        rows = [r[1:] if self.skip_bos else r for r in rows_per_prompt]
        if self.skip_bos:
            am[:, 0] = 0
        flat = am.reshape(-1).nonzero().squeeze(1)
        h, kwargs = self.cap.capture(enc)
        return {"h": h, "kw": {d: to_device(kwargs, d) for d in self.stage_devs},
                "flat": {d: flat.to(d, non_blocking=True) for d in self.enc_devs},
                "rowmap": np.concatenate(rows), "n_bos": 0 if self.skip_bos else len(rows_per_prompt)}

    def enqueue(self, preps):
        """Run all micro-batches of a wave through all stages (async); returns pending encode handles."""
        pend, state, S, M = {}, {}, len(self.stages), len(preps)
        with torch.no_grad():
            for t in range(M + S - 1):
                for s in range(S):
                    m = t - s
                    if not (0 <= m < M):
                        continue
                    d, Ls = self.stages[s]
                    with torch.cuda.device(d):
                        hcur = (preps[m]["h"] if s == 0 else state.pop(m)).to(d, non_blocking=True)
                        kwd = preps[m]["kw"][d]
                        for L in Ls:
                            out = self.tm.layers[L](hcur, **kwd)
                            hcur = out[0] if isinstance(out, tuple) else out
                            if L in self.lset:
                                x = hcur.reshape(-1, hcur.shape[-1]).index_select(0, preps[m]["flat"][hcur.device]).contiguous()
                                pend[(m, L)] = self.enc.launch(L, x, preps[m]["n_bos"])
                        if s < S - 1:
                            state[m] = hcur
        return pend

    def readback(self, pend, preps, emit):
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)
        for (m, L), h in sorted(pend.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            r, f, v = self.enc.collect(h)
            emit(L, preps[m]["rowmap"][r], f, v)

    def close(self):
        self.cap.remove()
