#!/usr/bin/env python3
"""Quality gate: teacher-forced decode of 512 real tokens against the CPU reference.

Every token goes through the same traced decode path that bench_real.py times. Logits at each position are compared with
the reference run of DeepSeek's own inference code. Metric: top-1 agreement where the reference is decisive (margin at
least 0.10), plus mean KL. Limits: decisive at least 0.98, KL at most 0.05.

Needs dsv41/dumps/eval/logits_nofakequant.pt, written by dsv41/ref/make_reference.py.

Run: ./run.sh bench/gate_512.py
"""
import json
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "dsv41"))
sys.path.insert(0, os.path.join(ROOT, "dsv41", "ref"))


def log(*a):
    print("[gate]", *a, flush=True)


def main():
    from live_model import Live40LEngine
    from evalmetrics import compare
    ref = torch.load(os.path.join(ROOT, "dsv41", "dumps", "eval", "logits_nofakequant.pt"), map_location="cpu", weights_only=False)
    ids = [int(x) for x in ref["ids"]]
    eng = Live40LEngine(max_ctx=1024)
    rows, times = [], []
    try:
        eng._reset_state_keep_trace(0)
        t0 = time.perf_counter()
        for i, t in enumerate(ids):
            ts = time.perf_counter()
            lg = eng.model.decode(t)
            times.append(time.perf_counter() - ts)
            rows.append(lg[-1].float().clone())
            if (i + 1) % 64 == 0:
                log(f"{i + 1}/{len(ids)} tokens, {time.perf_counter() - t0:.0f} s")
    finally:
        eng.close()
    m = compare(ref["logits"], torch.stack(rows), ids)
    srt = sorted(times)
    m.update(n_tokens=len(ids), ms_per_tok_median=round(srt[len(srt) // 2] * 1e3, 2), steps_over_1s=sum(1 for s in times if s > 1.0),
             wall_s=round(sum(times), 1), ok=int(m["top1_decisive"] >= 0.98 and m["kl_mean"] <= 0.05))
    log(json.dumps(m))
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(m, open(os.path.join(ROOT, "results", "gate_512.json"), "w"), indent=1)
    return 0 if m["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
