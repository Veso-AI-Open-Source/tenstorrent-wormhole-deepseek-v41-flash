#!/usr/bin/env python3
"""Real generation speed. Tokens come from the cards. Time is the host wall clock per token.

Gate: the reference continuation of "The capital of France is", checked over the first BENCH_TOKENS tokens when that is 16
or fewer. Longer runs report timing only, including the stalls when the decode traces are captured again.

Run: ./run.sh bench/bench_real.py
"""
import json
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "dsv41"))

GOLD = [11111, 16, 660, 270, 3584, 294, 3980, 14, 270, 5214, 6102, 769, 1047, 295, 1623, 12525]
PROMPT = [0, 671, 6102, 294, 8760, 344]


def log(*a):
    print("[bench]", *a, flush=True)


def gen(eng, ids, n):
    t0 = time.perf_counter()
    first, _, _ = eng.prefill_or_extend(list(ids), reset=True)
    t_pre = time.perf_counter() - t0
    out, steps = [int(first)], []
    for _ in range(n - 1):
        t = time.perf_counter()
        out.append(int(eng.step_decode()))
        steps.append(time.perf_counter() - t)
    return out, t_pre, steps


def main():
    from live_model import Live40LEngine
    n = int(os.environ.get("BENCH_TOKENS", "16"))
    t0 = time.perf_counter()
    eng = Live40LEngine(max_ctx=1024)
    load_s = time.perf_counter() - t0
    log(f"engine up in {load_s:.1f} s")
    res = {"tokens_requested": n, "load_s": round(load_s, 1), "prompt": PROMPT}
    try:
        for label in ("first", "repeat", "repeat2"):
            out, t_pre, steps = gen(eng, PROMPT, n)
            med = statistics.median(steps)
            r = dict(prefill_s=round(t_pre, 3), ms_per_tok_median=round(med * 1e3, 2), ms_per_tok_mean=round(sum(steps) / len(steps) * 1e3, 2),
                     ms_per_tok_max=round(max(steps) * 1e3, 2), steps_over_1s=sum(1 for s in steps if s > 1.0),
                     tok_s_median=round(1.0 / med, 2), tok_s_mean=round(len(steps) / sum(steps), 2))
            if n <= len(GOLD):
                r["tokens_ok"] = int(out == GOLD[:n])
            res[label] = r
            res[label + "_out"] = out
            log(label, json.dumps(r))
    finally:
        eng.close()
    path = os.environ.get("BENCH_OUT", os.path.join(ROOT, "results", f"bench_real_{n}.json"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(res, open(path, "w"), indent=1)
    log("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
