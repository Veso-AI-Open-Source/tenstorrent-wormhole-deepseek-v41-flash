"""DSpark drafter on the mesh (model.py DSparkBlock / forward_spec; the reference ships no loop).

Three draft blocks (`mtp.0-2`), each: window attention over a ring of the MAIN model's `main_x` keys, MoE of 128
resident experts (top-3 + shared), mHC. A draft pass reads main_hidden (the mean over hc copies of the attention
input of layers 37-39, concatenated) for the position just decoded, embeds [next token, noise x4] at positions
pos+1..pos+5 (bidirectional among the five, attending the whole ring), and the last block's head gives 5 x vocab
logits; the markov head (rank-256 bigram bias keyed by the previous draft) is applied sequentially on the host,
argmax by argmax, and a confidence per draft position comes from a tiny fp32 projection.

Eager v1: every step uploads the 5 embedding rows + rope tables + ring one-hot; the expert dispatch is per unique
expert with one-hot gather/scatter matmuls (exact). All state is per-block (ring); nothing is shared with the
main model except the head and embedding tables.
"""
import numpy as np
import torch
import ttnn
from mesh import to_dev, mm, rms_norm, shard, chip
from mhc import MHC
from attn import Attention, all_reduce_sum, NEG, WIN, H, D, TP
from moe import LIMIT, ROUTE_SCALE
import rope

BLOCK, NOISE, TARGETS, NEXP_D, TOPK_D, RANK = 5, 128799, (37, 38, 39), 128, 3, 256
DIM, HC = 5120, 4


class DraftBlock:
    def __init__(self, dev, R, m, store, sink_op=None, max_pos=4096, slots=None, slot_cat=None):
        self.dev, self.m, self.sink_op, self.slot_cat = dev, m, sink_op, slot_cat
        self.cat = slots.cat if slots is not None else None
        w = self.w = R.layer(m, prefix="mtp"); p = f"mtp.{m}"
        self.attn = Attention(dev, w, max_pos=max_pos)                       # window layer (compress_ratio 0)
        self.hc_attn = MHC(dev, R.torch(f"{p}.hc_attn_fn"), w["hc_attn_base"], w["hc_attn_scale"], sink_op=sink_op)
        self.hc_ffn = MHC(dev, R.torch(f"{p}.hc_ffn_fn"), w["hc_ffn_base"], w["hc_ffn_scale"], sink_op=sink_op)
        self.gate_w32, self.bias = w["gate_w32"], w["gate_bias"].float()   # [5120,128] fp32 on device, [128] host
        self.sh_w13 = ttnn.concat([w["sh_w1"], w["sh_w3"]], dim=3)
        # all 128 experts resident (bfp4 from the packed store, Lm{m}): 0.32 GB/chip per block
        self.experts = [{pj: ttnn.to_device(t, dev) for pj, t in store.get(f"Lm{m}", e).items()} for e in range(NEXP_D)]
        for sl in self.experts:
            sl["addrs"] = (int(sl["w1"].buffer_address()), int(sl["w3"].buffer_address()), int(sl["w2"].buffer_address()))

    # ---- attention: main key at `pos` joins the ring; the five drafts attend the ring + each other
    def attend(self, h, main_x, pos, ctx):
        """h [1,1,5,d] (normed draft stream), main_x [1,1,1,d] or None -> [1,1,5,d]."""
        a = self.attn
        if main_x is not None:
            from rope_cache_op import cache_write_multi
            main_kv = a._kv(main_x, None, None, apply_rope=False)
            cache_write_multi(a.win, main_kv, pos, 1, is_ring=True, cos=ctx.cs_main[0], sin=ctx.cs_main[1])
        _, q = a._q(h, *ctx.cs_draft)                                                        # [1,8,5,512] at pos+1..pos+5
        kv = a._kv(h, *ctx.cs_draft)                                                         # [1,1,5,512]
        return a._attend(q, ttnn.concat([a.win, kv], dim=2), ctx.mask, *ctx.cs_draft)        # [1,1,5,d] after all-reduce

    def seed(self, main_x, start_pos):
        """Prefill: the ring holds the main keys of the last 128 prompt positions (positions start_pos..)."""
        s = main_x.shape[2]
        cos, sin = rope.tables(self.dev, self.attn.angles[start_pos : start_pos + s])
        self.attn.ring_from_prefill(self.attn._kv(main_x, cos, sin), s, start_pos)

    # ---- MoE: 128 resident experts, top-3 (+ shared), 6-expert SlotCatOp groups (or exact one-hot fallback)
    def moe(self, h, active_rows=None, sc_host=None, gu_sh=None):
        if sc_host is None:
            sc_dev = mm(ttnn.typecast(h, ttnn.float32), self.gate_w32)
            sc_host = ttnn.from_device(sc_dev)
        sc = torch.nn.functional.softplus(ttnn.to_torch(ttnn.get_device_tensors(sc_host)[0]).float()[0, 0]).sqrt()
        idx = (sc + self.bias).topk(TOPK_D, dim=-1).indices
        wt = sc.gather(1, idx); wt = wt / (wt.sum(-1, keepdim=True) + 1e-20) * ROUTE_SCALE
        s = h.shape[2]
        if s <= 8 and self.slot_cat is not None and self.cat is not None:
            r_lim = s if active_rows is None else min(s, active_rows)
            idx_l = idx[:r_lim].tolist()
            used = sorted({e for row in idx_l for e in row})
            n_u = len(used)
            n_pad = -(-n_u // 6) * 6
            wt_pad = torch.zeros(n_pad, s, dtype=torch.bfloat16)
            e_to_u = {e: i for i, e in enumerate(used)}
            wt_bf16 = wt[:r_lim].to(torch.bfloat16)
            for r in range(r_lim):
                wt_pad[[e_to_u[e] for e in idx_l[r]], r] = wt_bf16[r]
            slot_list = [self.experts[e] for e in used] + [self.experts[used[-1]]] * (n_pad - n_u)
            acc = None
            for g0 in range(0, n_pad, 6):
                self.slot_cat.update_ctrl_multi(slot_list[g0 : g0 + 6], wt_pad[g0 : g0 + 6])
                self.slot_cat.run_from_ctrl("bfp4", self.cat)
                g = mm(h, self.cat["w1"])
                u = mm(h, self.cat["w3"])
                if g0 == 0:
                    if gu_sh is None:
                        gu_sh = mm(h, self.sh_w13)
                    h_routed, h_sh = self.slot_cat.swiglu_from_ctrl(g, u, gu_sh, s_rows=s)
                    acc = ttnn.add(mm(h_sh, self.w["sh_w2"]), mm(h_routed, self.cat["w2"]))
                else:
                    h_routed, _ = self.slot_cat.swiglu_from_ctrl(g, u, None, s_rows=s)
                    acc = ttnn.add(acc, mm(h_routed, self.cat["w2"]))
            return all_reduce_sum(acc)
        sh = {"w1": self.w["sh_w1"], "w3": self.w["sh_w3"], "w2": self.w["sh_w2"]}
        acc = self._ffn(h, sh)
        for e in idx.unique().tolist():
            rows, cols = torch.where(idx == e)
            n = len(rows); Dg = torch.zeros(32, s); Dg[torch.arange(n), rows] = 1.0
            Dw = torch.zeros(s, 32); Dw[rows, torch.arange(n)] = wt[rows, cols]
            ye = self._ffn(mm(to_dev(self.dev, Dg.view(1, 1, 32, s)), h), self.experts[e])
            acc = ttnn.add(acc, mm(to_dev(self.dev, Dw.view(1, 1, s, 32)), ye))
        return all_reduce_sum(acc)

    @staticmethod
    def _ffn(x, e):
        g = ttnn.clamp(mm(x, e["w1"]), max=LIMIT); u = ttnn.clamp(mm(x, e["w3"]), min=-LIMIT, max=LIMIT)
        return mm(ttnn.mul(ttnn.silu(g), u), e["w2"])

    def forward(self, x, pre_mix, main_x, pos, ctx, active_rows=None):
        """x [1,4,5,d], pre_mix [1,4,5,1] -> (x, next pre_mix), model.py Block.forward with DSparkAttention."""
        if self.sink_op is not None and x.shape[2] <= 8 and hasattr(self.sink_op, "pre_ms"):
            h_pre, ms = self.sink_op.pre_ms(x, pre_mix)
            a_pre, a_post, a_comb = self.hc_attn.mixes(x, ms=ms)
            h = rms_norm(h_pre, self.w["attn_norm"])
            x, h_pre, ms = self.sink_op.post_pre_ms(self.attend(h, main_x, pos, ctx), x, a_post, a_comb, a_pre)
            h = rms_norm(h_pre, self.w["ffn_norm"])
            sc_dev = mm(ttnn.typecast(h, ttnn.float32), self.gate_w32)
            sc_host = ttnn.from_device(sc_dev)
            f_pre, f_post, f_comb = self.hc_ffn.mixes(x, ms=ms)
            gu_sh = mm(h, self.sh_w13)
            return self.sink_op.post_only(self.moe(h, active_rows=active_rows, sc_host=sc_host, gu_sh=gu_sh), x, f_post, f_comb), f_pre
        a_pre, a_post, a_comb = self.hc_attn.mixes(x)
        h = rms_norm(MHC.pre(x, pre_mix), self.w["attn_norm"])
        x = MHC.post(self.attend(h, main_x, pos, ctx), x, a_post, a_comb)
        f_pre, f_post, f_comb = self.hc_ffn.mixes(x)
        h = rms_norm(MHC.pre(x, a_pre), self.w["ffn_norm"])
        return MHC.post(self.moe(h, active_rows=active_rows), x, f_post, f_comb), f_pre


class DraftCtx:
    """Per-step position data for the drafter: rope at pos (main key) and pos+1..pos+5 (drafts), ring one-hot,
    attention mask [1,1,5,133]: ring slot j visible iff j < min(128, pos+1) (the reference's topk_idxs), all five
    drafts visible to each other."""
    def __init__(self, dev, angles, pos, R=1, need_main=True):
        """R: rows of the main_x tensor (row 0 is the key at pos; extra rows are zero padding and get weight 0)."""
        if need_main:
            ang = angles[pos : pos + R]
            self.cs_main = rope.tables(dev, ang if ang.shape[0] == R else torch.cat([ang, torch.zeros(R - ang.shape[0], ang.shape[1])]))
        else:
            self.cs_main = None
        self.cs_draft = rope.tables(dev, angles[pos + 1 : pos + 1 + BLOCK])
        vis = torch.cat([torch.arange(WIN) < min(WIN, pos + 1), torch.ones(BLOCK, dtype=torch.bool)])
        self.mask = to_dev(dev, torch.where(vis, 0.0, NEG).view(1, 1, 1, WIN + BLOCK).expand(1, 1, BLOCK, WIN + BLOCK).contiguous())


class Drafter:
    def __init__(self, dev, R, store, embed, head, max_pos=4096, sink_op=None, slots=None, slot_cat=None):
        """embed: host bf16 [V,d] table (rows by id); head: the main model's Head (norm is the drafter's own)."""
        self.dev, self.embed, self.head, self.sink_op = dev, embed, head, sink_op
        self.blocks = [DraftBlock(dev, R, m, store, sink_op=sink_op, max_pos=max_pos, slots=slots, slot_cat=slot_cat) for m in range(3)]
        self.main_proj, self.main_norm = R.linear("mtp.0.main_proj.weight"), R.vec("mtp.0.main_norm.weight")
        self.norm = R.vec("mtp.2.norm.weight")
        self.mk_embed = R.torch("mtp.2.markov_head.embed.weight")                      # [V,256] host fp32
        self.mk_head = R.torch("mtp.2.markov_head.head.weight")                        # [V,256] host fp32
        self.mk_head_dev = R.linear("mtp.2.markov_head.head.weight", "col")            # [1,1,256,16160] per chip on device
        self.conf = R.torch("mtp.2.confidence_head.proj.weight").view(-1)             # [5376] host fp32
        self.angles = self.blocks[0].attn.angles

    def main_x(self, main_hidden):
        """main_hidden [1,1,s,3*d] on device -> main_x [1,1,s,d] (main_proj + main_norm)."""
        return rms_norm(mm(main_hidden, self.main_proj), self.main_norm)

    def seed(self, main_hidden, start_pos):
        mx = self.main_x(main_hidden)
        for b in self.blocks: b.seed(mx, start_pos)

    def rows(self, t, lo, hi):
        """t[:, :, lo:hi, :] as a fresh tensor with ZERO tile padding (a ttnn.slice leaves the padded rows undefined,
        and a padded-row NaN times a zero one-hot is NaN: it poisoned the ring through mm(P, kv))."""
        n = hi - lo; sel = torch.zeros(32, t.shape[2]); sel[torch.arange(n), torch.arange(lo, hi)] = 1.0
        return mm(to_dev(self.dev, sel.view(1, 1, 32, t.shape[2])), t)                  # [1,1,32,w]: rows 0..n-1 real, rest zero

    def append(self, main_hidden, start_pos, a=None):
        """Verified positions start_pos.. (the first a rows of main_hidden [1,1,s,3d]) join every block's ring in place."""
        from rope_cache_op import cache_write_multi
        s = main_hidden.shape[2]
        a = s if a is None else a
        mx = self.main_x(main_hidden)
        ang = self.angles[start_pos : start_pos + s]
        cos, sin = rope.tables(self.dev, ang if ang.shape[0] == s else torch.cat([ang, torch.zeros(s - ang.shape[0], ang.shape[1])]))
        for b in self.blocks:
            kv = b.attn._kv(mx, None, None, apply_rope=False)
            cache_write_multi(b.attn.win, kv, start_pos, a, is_ring=True, cos=cos, sin=sin)

    def snapshot(self):
        return [ttnn.clone(b.attn.win) for b in self.blocks]

    def restore(self, snap):
        for b, w in zip(self.blocks, snap): ttnn.copy(w, b.attn.win)

    def draft(self, next_token, main_hidden, pos, k=BLOCK, want_conf=True):
        """After the main model produced `next_token` at position pos (with main_hidden of pos, or None if already appended):
        k draft ids for positions pos+2..pos+1+k, their logits [k,V] (markov-biased) and confidences [k]. Reference forward_spec."""
        ids = [next_token] + [NOISE] * (BLOCK - 1)
        rows = self.embed[torch.as_tensor(ids, dtype=torch.long)].float().view(1, 1, BLOCK, DIM)
        x = ttnn.concat([to_dev(self.dev, rows)] * HC, dim=1)                              # [1,4,5,d]
        pre = torch.zeros(1, HC, BLOCK, 1); pre[:, 0] = 1.0; pre = to_dev(self.dev, pre)
        mx = self.main_x(main_hidden) if main_hidden is not None else None
        ctx = DraftCtx(self.dev, self.angles, pos, R=1 if main_hidden is None else main_hidden.shape[2], need_main=main_hidden is not None)
        for b in self.blocks:
            x, pre = b.forward(x, pre, mx, pos, ctx, active_rows=(k if b.m == 2 else None))
        if self.sink_op is not None and hasattr(self.sink_op, "pre_ms"):
            hcol = self.sink_op.pre_ms(x, pre)[0]
        else:
            hcol = MHC.pre(x, pre)                                                          # [1,1,5,d] (pre-norm: the confidence input)
        me0 = self.mk_embed[next_token]
        me_pad = torch.zeros(1, 1, BLOCK, RANK)
        me_pad[0, 0, 0] = me0
        lg = ttnn.add(mm(rms_norm(hcol, self.norm), self.head.head), mm(to_dev(self.dev, me_pad), self.mk_head_dev))
        if k <= 8:
            from rope_cache_op import untilize_head_1
            lg = untilize_head_1(self.dev, lg, s_rows=k)
        logits = torch.cat([ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(ttnn.from_device(lg))], dim=-1)[0, 0]   # [k,V]
        out = [next_token, int(logits[0].argmax())]
        embeds = [me0]
        for i in range(1, k):
            me = self.mk_embed[out[i]]                                                      # [256]
            logits[i] += self.mk_head @ me
            embeds.append(me); out.append(int(logits[i].argmax()))
        if not want_conf:
            return out[1:], logits[:k], None
        hid = chip(hcol)[0, 0]                                                              # [5,d]
        me = torch.stack(embeds)                                                            # [k,256]
        conf = torch.cat([hid[:k], me], dim=-1) @ self.conf                                 # [k]
        return out[1:], logits[:k], conf
