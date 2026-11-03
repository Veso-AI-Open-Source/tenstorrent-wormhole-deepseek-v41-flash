#!/usr/bin/env python3
"""True device cost of each sublayer group of a real layer. Each group runs from real engine buffers in its own trace,
replayed N times with one sync at the end. No per-replay sync, so the sync call cannot distort it.

Run: ./run.sh bench/bench_layer.py
"""
import json
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "dsv41"))
N = 100
PROMPT = [0, 671, 6102, 294, 8760, 344]


def main():
    import ttnn
    from live_model import Live40LEngine
    eng = Live40LEngine(max_ctx=1024)
    m, dev = eng.model, eng.dev
    eng.prefill_or_extend(list(PROMPT), reset=True)
    for _ in range(3):
        eng.step_decode()
    tr = m._trace
    from mesh import rms_norm, mm
    from attn import all_reduce_sum
    sink, ctx, pos, shared = m.sink_op, tr.ctx, m.pos, m.shared
    x, pre = tr.x_in, tr.pre_in
    res = {}

    def measure(label, fn):
        ttnn.synchronize_device(dev)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        fn()
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        ts = []
        for _ in range(3):
            ttnn.synchronize_device(dev)
            t = time.perf_counter()
            for _ in range(N):
                ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(dev)
            ts.append((time.perf_counter() - t) / N)
        v = round(statistics.median(ts) * 1e6, 1)
        ttnn.release_trace(dev, tid)
        res[label] = v
        print(f"[layer] {label:38s} {v:8.1f} us", flush=True)
        return v

    for L in (3, 0):
        l = m.layers[L]
        w = l.w
        fmt = m.store.fmt(L)
        cat = m.slots[fmt].cat
        h_pre, ms = sink.pre_ms(x, pre)
        a_pre, a_post, a_comb = l.hc_attn.mixes(x, ms=ms)
        h = rms_norm(h_pre, w["attn_norm"])
        y = l.attn.decode_window(h, pos, ctx) if l.mode is None else l.attn.decode(h, pos, shared, ctx)
        x2, h_pre2, ms2 = sink.post_pre_ms(y, x, a_post, a_comb, a_pre)
        h2 = rms_norm(h_pre2, w["ffn_norm"])
        p = f"L{L} "
        t = {}
        t["pre_ms"] = measure(p + "A pre_ms (mHC pre + stats)", lambda: sink.pre_ms(x, pre))
        t["mix_a"] = measure(p + "A attention mix (fn4, Sinkhorn)", lambda: l.hc_attn.mixes(x, ms=ms))
        t["rms_a"] = measure(p + "A rms_norm attn", lambda: rms_norm(h_pre, w["attn_norm"]))
        t["attn"] = measure(p + "A attention, whole", lambda: l.attn.decode_window(h, pos, ctx) if l.mode is None else l.attn.decode(h, pos, shared, ctx))
        t["post_pre"] = measure(p + "A post_pre_ms", lambda: sink.post_pre_ms(y, x, a_post, a_comb, a_pre))
        t["rms_f"] = measure(p + "A rms_norm ffn", lambda: rms_norm(h_pre2, w["ffn_norm"]))
        t["scores"] = measure(p + "A router scores", lambda: l.moe.scores(h2))
        t["slot_cat"] = measure(p + "B slot gather", lambda: m.slot_cat.run_from_ctrl(fmt, cat))
        t["moe"] = measure(p + "B MoE, whole", lambda: l.moe.decode(h2, None, None, cat=cat, slot_cat=m.slot_cat))
        t["mix_f"] = measure(p + "B ffn mix", lambda: l.hc_ffn.mixes(x2, ms=ms2))
        res[p + "group_sum"] = round(sum(t.values()), 1)
        print(f"[layer] {p}group sum {sum(t.values()):.1f} us, times 40 = {sum(t.values()) * 40 / 1e3:.1f} ms", flush=True)
        if L == 3:
            cs = ctx.cs_csa if hasattr(ctx, "cs_csa") else ctx.cs_win
            measure(p + "  in attn: q (wq_a, norm, wq_b, rope)", lambda: l.attn._q(h, *cs))
            measure(p + "  in attn: kv (wkv, norm)", lambda: l.attn._kv(h, None, None, apply_rope=False))
            q_ = mm(h, w["wq_a"])
            measure(p + "  in MoE: w1 w3 routed", lambda: (mm(h2, cat["w1"]), mm(h2, cat["w3"])))
            g_, u_ = mm(h2, cat["w1"]), mm(h2, cat["w3"])
            gs_ = mm(h2, l.moe.sh_w13)
            measure(p + "  in MoE: shared w13", lambda: mm(h2, l.moe.sh_w13))
            measure(p + "  in MoE: swiglu", lambda: m.slot_cat.swiglu_from_ctrl(g_, u_, gs_))
            hr_, hs_ = m.slot_cat.swiglu_from_ctrl(g_, u_, gs_)
            measure(p + "  in MoE: w2 routed", lambda: mm(hr_, cat["w2"]))
            measure(p + "  in MoE: w2 shared", lambda: mm(hs_, w["sh_w2"]))
            part = mm(hs_, w["sh_w2"])
            measure(p + "  in MoE: all_reduce_sum", lambda: all_reduce_sum(part))
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "results", "bench_layer.json"), "w"), indent=1)
    eng.close()


if __name__ == "__main__":
    sys.exit(main())
