#!/usr/bin/env python3
"""Cost of one stock op inside a replayed trace on the 8-chip mesh.

Each op gets its own trace, replayed N times with one device sync at the end. A sync around every replay is not usable: the
sync itself takes about 400 us and varies enough to produce negative op times. Real weight shapes, random values, timing only.

Run: ./run.sh bench/bench_ops.py
"""
import json
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "dsv41", "tt"))
import torch  # noqa: E402
import ttnn  # noqa: E402
from mesh import open_mesh, close_mesh, CK, mm  # noqa: E402

N = 200


def main():
    dev = open_mesh(trace_region_mb=64)
    rep = ttnn.ReplicateTensorToMesh(dev)
    z = lambda *s, dt=ttnn.bfloat16: ttnn.from_torch(torch.randn(*s), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev, mesh_mapper=rep)
    res = {}

    def measure(label, fn):
        fn()
        ttnn.synchronize_device(dev)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        fn()
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        ts = []
        for _ in range(4):
            ttnn.synchronize_device(dev)
            t = time.perf_counter()
            for _ in range(N):
                ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(dev)
            ts.append((time.perf_counter() - t) / N)
        med = statistics.median(ts) * 1e6
        res[label] = round(med, 1)
        print(f"[ops] {label:34s} {med:8.1f} us  (spread {min(ts) * 1e6:.1f} to {max(ts) * 1e6:.1f})", flush=True)
        ttnn.release_trace(dev, tid)

    for name, k, j in (("wq_a 5120->1280", 5120, 1280), ("wkv 5120->512", 5120, 512), ("wq_b 1280->4096", 1280, 4096),
                       ("wo_b 1024->5120", 1024, 5120), ("sh_w2 288->5120", 288, 5120)):
        x, W = z(1, 1, 1, k), z(1, 1, k, j)
        measure(f"matmul {name}", lambda: mm(x, W))
    x, a = z(1, 1, 1, 5120), z(1, 1, 1, 5120)
    small, small2 = z(1, 1, 1, 32), z(1, 1, 1, 32)
    measure("typecast", lambda: ttnn.typecast(x, ttnn.float32))
    measure("add", lambda: ttnn.add(x, a))
    measure("rms_norm 5120", lambda: ttnn.rms_norm(x, weight=a, epsilon=1e-20, compute_kernel_config=CK))
    measure("all_gather + sum", lambda: ttnn.sum(ttnn.all_gather(x, dim=1), dim=1, keepdim=True))
    measure("copy 5120", lambda: ttnn.copy(x, a))
    measure("copy 1 tile", lambda: ttnn.copy(small, small2))
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "results", "bench_ops.json"), "w"), indent=1)
    close_mesh(dev)
    return 0


if __name__ == "__main__":
    sys.exit(main())
