"""
saepipe/io.py — output of sparse codes. [generic]
Format per layer L: rows (int32 global token index), feats (int32 feature id), vals (float32 activation),
either one file layer_<L>.npz (SparseSink) or streamed shards layer_<L>/part_<k>.npz (ShardWriter).
"""
import time
from collections import defaultdict
from pathlib import Path
import numpy as np


class SparseSink:
    """Collects (rows, feats, vals) per layer as CPU numpy chunks."""
    def __init__(self):
        self.chunks = defaultdict(list)

    def add(self, layer, rows, feats, vals):
        self.chunks[layer].append((np.asarray(rows, dtype=np.int32),
                                   np.asarray(feats, dtype=np.int32),
                                   np.asarray(vals, dtype=np.float32)))

    def nnz(self, layer):
        return sum(len(c[0]) for c in self.chunks[layer])

    def save(self, run_dir, layer):
        ch = self.chunks.pop(layer, [])
        if ch:
            r, f, v = (np.concatenate(x) for x in zip(*ch))
        else:
            r = f = np.zeros(0, np.int32); v = np.zeros(0, np.float32)
        o = np.lexsort((f, r))
        np.savez(run_dir / f"layer_{layer}.npz", rows=r[o], feats=f[o], vals=v[o])
        return len(r)

    def drop(self, layer):
        n = self.nnz(layer)
        self.chunks.pop(layer, None)
        return n


class ShardWriter:
    """Background writer: per-layer shards runs/<run>/layer_<L>/part_<k>.npz written by a thread while
    the GPUs keep computing (streaming output; memory holds only unwritten waves)."""
    def __init__(self, run_dir, n_threads=4):
        import queue, threading
        self.run_dir, self.q = run_dir, queue.Queue(maxsize=8 * n_threads)
        self.parts, self.nnz, self.busy, self.bytes = defaultdict(int), defaultdict(int), 0.0, 0
        self.lock = threading.Lock()
        self.ts = [threading.Thread(target=self._work, daemon=True) for _ in range(n_threads)]
        for t in self.ts:
            t.start()

    def put(self, layer, rows, feats, vals):
        self.nnz[layer] += len(rows)
        self.q.put((layer, rows, feats, vals))

    def _work(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            L, r, f, v = item
            t0 = time.perf_counter()
            d = self.run_dir / f"layer_{L}"
            with self.lock:
                d.mkdir(exist_ok=True)
                k = self.parts[L]; self.parts[L] += 1
            fn = d / f"part_{k:05d}.npz"
            np.savez(fn, rows=r, feats=f, vals=v)
            with self.lock:
                self.bytes += fn.stat().st_size
                self.busy += time.perf_counter() - t0

    def close(self):
        for _ in self.ts:
            self.q.put(None)
        for t in self.ts:
            t.join()


def load_layer(run_dir, layer):
    """(rows, feats, vals) sorted by (row, feat), from layer_<L>.npz or streamed shards layer_<L>/part_*.npz."""
    run_dir = Path(run_dir)
    f = run_dir / f"layer_{layer}.npz"
    if f.exists():
        z = np.load(f)
        return z["rows"], z["feats"], z["vals"]
    parts = sorted((run_dir / f"layer_{layer}").glob("part_*.npz"))
    zs = [np.load(p) for p in parts]
    r = np.concatenate([z["rows"] for z in zs]); fe = np.concatenate([z["feats"] for z in zs])
    v = np.concatenate([z["vals"] for z in zs])
    o = np.lexsort((fe, r))
    return r[o], fe[o], v[o]


