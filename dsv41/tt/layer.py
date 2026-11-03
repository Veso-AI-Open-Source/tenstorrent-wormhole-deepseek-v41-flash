"""One transformer block on the mesh (prefill), wiring mHC around attention and MoE exactly as
model.py Block.forward does: the coefficients a sublayer computes are consumed by the NEXT sublayer."""
import ttnn
from mesh import to_dev, rms_norm
from mhc import MHC
from attn import Attention
from csa import CSA
from moe import MoE

KV_SOURCES, INDEX_SOURCES = (2, 8, 14, 20), (2, 8, 14, 20, 24, 28, 32, 36)


def layer_mode(L):
    ratio = 0 if L < 2 else (2 if L < 20 else 1)
    mode = None if ratio == 0 else ("full" if L in KV_SOURCES else "reindex" if L in INDEX_SOURCES else "reuse")
    return ratio, mode


class Layer:
    def __init__(self, dev, R, L, max_pos=4096, slots=None, sink_op=None, slot_cat=None):
        self.dev, self.L, self.sink_op = dev, L, sink_op
        w = self.w = R.layer(L); p = f"layers.{L}"
        self.ratio, self.mode = layer_mode(L)
        self.attn = Attention(dev, w, max_pos=max_pos) if self.mode is None else CSA(dev, w, self.ratio, self.mode, max_pos=max_pos)
        self.moe = MoE(dev, w, slots, slot_cat=slot_cat)
        self.hc_attn = MHC(dev, R.torch(f"{p}.hc_attn_fn"), w["hc_attn_base"], w["hc_attn_scale"], sink_op=sink_op)
        self.hc_ffn = MHC(dev, R.torch(f"{p}.hc_ffn_fn"), w["hc_ffn_base"], w["hc_ffn_scale"], sink_op=sink_op)

    def _sub(self, x, mix, norm_w):
        return rms_norm(MHC.pre(x, mix), norm_w)                                   # [1,1,s,d]

    def prefill(self, x, pre_mix, shared, get_expert, pool=None, start_pos=0, kv_only=False, kv_ready=False, history=True, image_mask=None):
        """x [1,4,s,d], pre_mix [1,4,s,1] -> (x_out, pre_mix for the next layer)."""
        a_pre, a_post, a_comb = self.hc_attn.mixes(x)
        h = self._sub(x, pre_mix, self.w["attn_norm"])
        if kv_only:                                                   # CED: feed the decoder's KV source and stop
            self.attn.prefill(h, shared, start_pos=start_pos, kv_only=True)
            return None, None
        h = (self.attn.prefill_window(h, start_pos=start_pos, cache=shared.__dict__.setdefault("cache", {})) if self.mode is None
             else self.attn.prefill(h, shared, start_pos=start_pos, kv_ready=kv_ready, history=history))
        x = MHC.post(h, x, a_post, a_comb)
        f_pre, f_post, f_comb = self.hc_ffn.mixes(x)
        h = self._sub(x, a_pre, self.w["ffn_norm"])
        idx, wt, _ = self.moe.route(h, image_mask)
        if pool is not None and pool.n >= 384:
            h, self.moe_stats = self.moe.prefill_batched(h, idx, wt, self.L, pool, get_expert)
        else:
            if hasattr(get_expert, "__self__") and hasattr(get_expert.__self__, "ensure_packed_many"):
                get_expert.__self__.ensure_packed_many([(self.L, int(e)) for e in idx.unique().tolist()])
            h = self.moe.prefill(h, idx, wt, lambda e: get_expert(self.L, e))
        return MHC.post(h, x, f_post, f_comb), f_pre

    def decode_multi(self, x, pre_mix, pos, shared, get_expert, ctxm, pool, want_mh=False, prof=None):
        """s tokens at pos.. through the decode semantics with in-place state (DSpark verify). Returns
        (x_out, next pre_mix, main_hidden [1,1,s,d] or None). prof: dev dict of per-component seconds (adds syncs)."""
        import time
        def tick(tag):
            if prof is not None:
                ttnn.synchronize_device(self.dev); t = time.perf_counter(); prof[tag] += t - tick.t; tick.t = t
        if prof is not None: ttnn.synchronize_device(self.dev); tick.t = time.perf_counter()
        mh = ttnn.mean(x, dim=1, keepdim=True) if want_mh else None
        if self.sink_op is not None and x.shape[2] <= 8 and hasattr(self.sink_op, "pre_ms"):
            from mesh import mm
            x_in = x if want_mh else None
            h_pre, ms = self.sink_op.pre_ms(x, pre_mix)
            a_pre, a_post, a_comb = self.hc_attn.mixes(x, ms=ms)
            h = rms_norm(h_pre, self.w["attn_norm"]); tick("mhc+norm")
            h = self.attn.decode_multi(h, pos, ctxm) if self.mode is None else self.attn.decode_multi(h, pos, shared, ctxm); tick("attention")
            x, h_pre, ms = self.sink_op.post_pre_ms(h, x, a_post, a_comb, a_pre)
            h = rms_norm(h_pre, self.w["ffn_norm"]); tick("mhc+norm")
            sc_dev = self.moe.scores(h)
            sc_host = ttnn.from_device(sc_dev)
            f_pre, f_post, f_comb = self.hc_ffn.mixes(x, ms=ms)
            gu_sh = mm(h, self.moe.sh_w13)
            if want_mh: mh = ttnn.mean(x_in, dim=1, keepdim=True)
            tick("mhc+norm")
            idx, wt, _ = self.moe.pick_from_host(sc_host); tick("router")
            h = self.moe.verify_batched(h, idx, wt, self.L, pool, get_expert, prof=prof, gu_sh=gu_sh); tick("moe compute")
            out = self.sink_op.post_only(h, x, f_post, f_comb); tick("mhc+norm")
            return out, f_pre, mh
        a_pre, a_post, a_comb = self.hc_attn.mixes(x)
        h = self._sub(x, pre_mix, self.w["attn_norm"]); tick("mhc+norm")
        h = self.attn.decode_multi(h, pos, ctxm) if self.mode is None else self.attn.decode_multi(h, pos, shared, ctxm); tick("attention")
        x = MHC.post(h, x, a_post, a_comb)
        f_pre, f_post, f_comb = self.hc_ffn.mixes(x)
        h = self._sub(x, a_pre, self.w["ffn_norm"]); tick("mhc+norm")
        idx, wt, _ = self.moe.route(h); tick("router")
        h = self.moe.verify_batched(h, idx, wt, self.L, pool, get_expert, prof=prof); tick("moe compute")
        out = MHC.post(h, x, f_post, f_comb); tick("mhc+norm")
        return out, f_pre, mh

    def decode(self, x, pre_mix, pos, shared, get_expert, ctx=None, pool=None):
        """One token: x [1,4,1,d], pre_mix [1,4,1,1] -> (x_out, next pre_mix)."""
        return self.decode_b(self.decode_a(x, pre_mix, pos, shared, ctx), get_expert, pool)

    # The step is split where the host must look: everything up to the router scores is pure device work with
    # no data-dependent shape or branch (decode_a: captured as a trace in s18), then the host picks the six
    # experts and the expert half runs eagerly on whichever pool slots hold them (decode_b).
    def decode_a(self, x, pre_mix, pos, shared, ctx=None, want_mh=False, defer_ffn_mix=False):
        mh = ttnn.mean(x, dim=1, keepdim=True) if want_mh else None      # DSpark: mean over the hc copies of this layer's input
        if self.sink_op is not None and hasattr(self.sink_op, "pre_ms"):
            h_pre, ms = self.sink_op.pre_ms(x, pre_mix)
            a_pre, a_post, a_comb = self.hc_attn.mixes(x, ms=ms)
            h = rms_norm(h_pre, self.w["attn_norm"])
            h = self.attn.decode_window(h, pos, ctx) if self.mode is None else self.attn.decode(h, pos, shared, ctx)
            x, h_pre, ms = self.sink_op.post_pre_ms(h, x, a_post, a_comb, a_pre)
            h = rms_norm(h_pre, self.w["ffn_norm"])
            if defer_ffn_mix:
                out = {"h": h, "sc": self.moe.scores(h)}
                if mh is not None: out["mh"] = mh
                return out
            f_pre, f_post, f_comb = self.hc_ffn.mixes(x, ms=ms)
        else:
            a_pre, a_post, a_comb = self.hc_attn.mixes(x)
            h = self._sub(x, pre_mix, self.w["attn_norm"])
            h = self.attn.decode_window(h, pos, ctx) if self.mode is None else self.attn.decode(h, pos, shared, ctx)
            x = MHC.post(h, x, a_post, a_comb)
            h = self._sub(x, a_pre, self.w["ffn_norm"])
            f_pre, f_post, f_comb = self.hc_ffn.mixes(x)
        out = {"h": h, "sc": self.moe.scores(h), "x": x, "f_pre": f_pre, "f_post": f_post, "f_comb": f_comb}
        if mh is not None: out["mh"] = mh
        return out

    def decode_b(self, a, get_expert, pool=None):
        idx, wt, _ = self.moe.pick(a["sc"])
        if pool is not None:
            h = self.moe.decode(a["h"], None, wt[0], slots=pool.ensure([(self.L, int(e)) for e in idx[0]], get_expert))
        else:
            h = self.moe.decode(a["h"], [get_expert(self.L, int(e)) for e in idx[0]], wt[0])
        if self.sink_op is not None and hasattr(self.sink_op, "post_only"):
            return self.sink_op.post_only(h, a["x"], a["f_post"], a["f_comb"]), a["f_pre"]
        return MHC.post(h, a["x"], a["f_post"], a["f_comb"]), a["f_pre"]
