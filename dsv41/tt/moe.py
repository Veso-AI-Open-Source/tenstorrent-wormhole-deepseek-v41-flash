"""MoE on the mesh. The router matmul runs on the device. Softplus, sqrt, top-6, and the weights run on the
host, because the host has to page the chosen experts in anyway ("stream the six"). Experts are TP-8: w1/w3 column-parallel
(2304 -> 288 per chip), w2 row-parallel, one all-reduce for routed + shared together.

Reference: model.py Gate / Expert / MoE. Scores = sqrt(softplus(x @ Wg)); the bias picks experts but
does not weight them; weights renormalize over the chosen six and scale by 1.5; SwiGLU clamps: gate
<= 10, up in [-10, 10]; routing weight is applied BEFORE w2.
"""
import numpy as np
import torch
import ttnn
from mesh import to_dev, shard, rep, mm, chip
from attn import all_reduce_sum

TOPK, NEXP, LIMIT, ROUTE_SCALE = 6, 384, 10.0, 1.5


def expert_to_mesh(dev, ck, layer, e, dtype, prefix="layers"):
    """One expert from the checkpoint -> three mesh tensors in matmul layout (stock converter)."""
    out = {}
    for pj, how in (("w1", 3), ("w3", 3), ("w2", 2)):
        w = torch.from_numpy(ck.expert_proj(layer, e, pj, prefix)[0].copy()).T.contiguous()      # [in, out]
        out[pj] = to_dev(dev, w, shard(dev, how), dtype)
    return out


def host_expert(dev, ck, layer, e, dtype, prefix="layers"):
    """Same, but left on the HOST, pre-tiled and pre-sharded: the unit that gets streamed."""
    out = {}
    for pj, how in (("w1", 3), ("w3", 3), ("w2", 2)):
        w = torch.from_numpy(ck.expert_proj(layer, e, pj, prefix)[0].copy()).T.contiguous()
        out[pj] = ttnn.from_torch(w[None, None], dtype=dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=shard(dev, how))
    return out


class Slots:
    """SIX FIXED expert buffers at fixed device addresses, shared by every layer of one format: the
    whole-tensor H2D targets of stream-the-six (and what a decode trace will point at), plus pre-allocated
    concatenated 6-expert buffers `cat` for zero-concat 3-matmul traced decode."""
    def __init__(self, dev, dtype):
        z = lambda a, b, how: to_dev(dev, torch.zeros(1, 1, a, b), shard(dev, how), dtype)
        self.s = [{"w1": z(5120, 2304, 3), "w3": z(5120, 2304, 3), "w2": z(2304, 5120, 2)} for _ in range(TOPK)]
        for sl in self.s:
            sl["addrs"] = (int(sl["w1"].buffer_address()), int(sl["w3"].buffer_address()), int(sl["w2"].buffer_address()))
        self.cat = {"w1": z(5120, TOPK * 2304, 3), "w3": z(5120, TOPK * 2304, 3), "w2": z(TOPK * 2304, 5120, 2)}
        self.cat["addrs"] = [int(self.cat["w1"].buffer_address()), int(self.cat["w3"].buffer_address()), int(self.cat["w2"].buffer_address())]


class ExpertPool:
    """N resident expert slots of ONE format, LRU over (layer, expert). A slot is three whole tensors at
    fixed addresses, so a miss is three whole-tensor H2D copies (fine for bfp4, no slot-write op needed) and
    a hit is free. This is the dynamic version of the "hot store": it follows the session instead of a
    frequency table, and needs no sparse_matmul."""
    def __init__(self, dev, dtype, n):
        from collections import OrderedDict
        self.dev, self.n = dev, n
        zero = {pj: ttnn.from_torch(torch.zeros(1, 1, *shp), dtype=dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=shard(dev, how))
                for pj, shp, how in (("w1", (5120, 2304), 3), ("w3", (5120, 2304), 3), ("w2", (2304, 5120), 2))}
        self.slot = [{pj: ttnn.to_device(zero[pj], dev) for pj in zero} for _ in range(n)]      # zeros converted once, uploaded n times
        for sl in self.slot:
            sl["addrs"] = (int(sl["w1"].buffer_address()), int(sl["w3"].buffer_address()), int(sl["w2"].buffer_address()))
        self.where, self.lru, self.hits, self.misses = {}, OrderedDict(), 0, 0

    def ensure(self, keys, get_host):
        """keys: this layer's six (layer, expert). Returns their six slots, paging in the missing ones."""
        where, slot, lru = self.where, self.slot, self.lru
        if all(k in where for k in keys):
            self.hits += len(keys)
            if len(where) >= self.n:
                for k in keys:
                    lru.move_to_end(k)
            return [slot[where[k]] for k in keys]
        need = set(keys); out = []
        missing = [k for k in need if k not in where]
        if missing and hasattr(get_host, "__self__") and hasattr(get_host.__self__, "ensure_packed_many"):
            get_host.__self__.ensure_packed_many(missing)
        for k in keys:
            sid = where.get(k)
            if sid is None:
                self.misses += 1
                if len(where) < self.n:
                    sid = len(where)
                else:
                    victim = next(o for o in lru if o not in need)
                    sid = where.pop(victim); del lru[victim]
                h = get_host(*k)
                for pj in ("w1", "w3", "w2"):
                    ttnn.copy_host_to_device_tensor(h[pj], slot[sid][pj])
                where[k] = sid
            else:
                self.hits += 1
            lru[k] = None; lru.move_to_end(k)
            out.append(slot[sid])
        return out


class MoE:
    def __init__(self, dev, w, slots: Slots = None, slot_dtype=ttnn.bfloat8_b, slot_cat=None):
        self.dev, self.w, self.slot_cat = dev, w, slot_cat
        self.bias = w["gate_bias"].float()                                              # [384] host
        slot_obj = slots or Slots(dev, slot_dtype)
        self.slots = slot_obj.s
        self.cat = slot_obj.cat
        self.fmt = "bfp4" if self.cat["w1"].dtype == ttnn.bfloat4_b else "bfp8"
        self.sh_w13 = ttnn.concat([w["sh_w1"], w["sh_w3"]], dim=3)                      # [1,1,5120,576] per chip

    # ---- router
    def scores(self, x):
        """x [1,1,s,5120] -> x @ Wg [1,1,s,384] float32 on device (softplus+sqrt applied on host in pick())."""
        # ROUTER IN FLOAT32. Measured on real activations: bf16 picks a different six on 7% of tokens
        # (the 6th/7th biased scores differ by as little as 1e-4), float32 on 1%. 2M weights: free.
        return mm(ttnn.typecast(x, ttnn.float32), self.w["gate_w32"])

    def route(self, x, image_mask=None):
        """Device scores -> host picks the six (index selection) -> (idx [s,6], weights [s,6]).
        image_mask [s] bool: image-span tokens select with the VL correction bias (reference Gate.forward)."""
        idx, wt, sc = self.pick(self.scores(x), image_mask)
        return idx, wt, sc

    def pick_from_host(self, scores_host, image_mask=None):
        sc = torch.nn.functional.softplus(ttnn.to_torch(ttnn.get_device_tensors(scores_host)[0]).float()[0, 0]).sqrt()
        bias = self.bias if image_mask is None else torch.where(image_mask[:, None], self.w["gate_bias_vl"].float(), self.bias)
        idx = (sc + bias).topk(TOPK, dim=-1).indices
        wt = sc.gather(1, idx)
        return idx, wt / (wt.sum(-1, keepdim=True) + 1e-20) * ROUTE_SCALE, sc

    def pick(self, scores_dev, image_mask=None):
        """Host half of the router: device scores [1,1,s,384] -> (idx [s,6], normalized weights [s,6], scores)."""
        return self.pick_from_host(ttnn.from_device(scores_dev), image_mask)

    # ---- experts
    @staticmethod
    def _ffn(x, e, wt=None):
        g = ttnn.clamp(mm(x, e["w1"]), max=LIMIT)
        u = ttnn.clamp(mm(x, e["w3"]), min=-LIMIT, max=LIMIT)
        h = ttnn.mul(ttnn.silu(g), u)
        if wt is not None:
            h = ttnn.mul(h, wt)
        return mm(h, e["w2"])                                                           # partial (row-parallel)

    def shared(self, x):
        return self._ffn(x, {"w1": self.w["sh_w1"], "w3": self.w["sh_w3"], "w2": self.w["sh_w2"]})

    def stream(self, slot, host_e, cq_id=0):
        for pj in ("w1", "w3", "w2"):
            ttnn.copy_host_to_device_tensor(host_e[pj], self.slots[slot][pj], cq_id=cq_id)

    def decode(self, x, host_experts, wts, slots=None, wts_dev=None, cat=None, slot_cat=None):
        """x [1,1,1,5120]; wts: the six routing weights (host list, or `wts_dev` [1,1,1,1728] or [1,6,1,1] already on
        device: the traced path keeps it at a fixed address). Either `cat` (pre-concatenated 6-expert buffers populated
        by SlotCatOp), `slots` (six resident pool slots, already holding the experts) or `host_experts` (streamed into
        the six fixed slots first).
        All six routed experts are concatenated into 54-tile-wide matrices so w1, w3, and w2 each run as a single
        full-grid matmul instead of 6 narrow 9-tile matmuls + 6 slices + 6 adds."""
        if cat is not None:
            w1_cat, w3_cat, w2_cat = cat["w1"], cat["w3"], cat["w2"]
            if slot_cat is not None:
                g = mm(x, w1_cat)                                                          # [1,1,1,1728]
                u = mm(x, w3_cat)                                                          # [1,1,1,1728]
                gu_sh = mm(x, self.sh_w13)                                                 # [1,1,1,576]
                h_routed, h_sh = slot_cat.swiglu_from_ctrl(g, u, gu_sh)
                return all_reduce_sum(ttnn.add(mm(h_sh, self.w["sh_w2"]), mm(h_routed, w2_cat)))
        else:
            if slots is None:
                for k in range(TOPK):
                    self.stream(k, host_experts[k])
                slots = self.slots
            w1_cat = ttnn.concat([slots[k]["w1"] for k in range(TOPK)], dim=3)             # [1,1,5120,1728] per chip
            w3_cat = ttnn.concat([slots[k]["w3"] for k in range(TOPK)], dim=3)             # [1,1,5120,1728] per chip
            w2_cat = ttnn.concat([slots[k]["w2"] for k in range(TOPK)], dim=2)             # [1,1,1728,5120] per chip
        g = mm(x, w1_cat)                                                                  # [1,1,1,1728]
        u = mm(x, w3_cat)                                                                  # [1,1,1,1728]
        h = ttnn.mul(ttnn.silu(ttnn.clamp(g, max=LIMIT)), ttnn.clamp(u, min=-LIMIT, max=LIMIT))
        if wts_dev is None:
            wts_dev = to_dev(self.dev, torch.as_tensor(wts, dtype=torch.float32).repeat_interleave(288).view(1, 1, 1, TOPK * 288))
        elif wts_dev.shape[1] == TOPK:
            h = ttnn.reshape(ttnn.mul(ttnn.reshape(h, (1, TOPK, 1, 288)), wts_dev), (1, 1, 1, TOPK * 288))
            return all_reduce_sum(ttnn.add(self.shared(x), mm(h, w2_cat)))
        h = ttnn.mul(h, wts_dev)
        return all_reduce_sum(ttnn.add(self.shared(x), mm(h, w2_cat)))

    def prefill_batched(self, x, idx, wt, layer, pool, get_host, C=None):
        """Whole-layer batched MoE for prefill. x [1,1,s,5120]; idx/wt [s,6] (host).
        Routing inside one document is wildly uneven (measured: one expert took 207 of 512 tokens, 26x the
        mean), so capacity is NOT per expert: every expert gets ceil(n_e / C) VIRTUAL buckets of C rows that
        share its weights (the concat list simply repeats that expert's slot tensor). Rows = s*6 + padding,
        independent of imbalance; the bucket count is rounded up to a multiple of 32 to bound program shapes.
        Flow: stream used experts into pool slots (slot e = expert e) -> ttnn.embedding gathers tokens into
        buckets -> three batched matmuls -> each token gathers its six results back -> weighted sum, BEFORE
        the all-reduce (the row-parallel partial is then s x 5120, not rows x 5120)."""
        s, d, E = x.shape[2], x.shape[3], NEXP
        if s <= 8 and self.slot_cat is not None and self.cat is not None:
            return self.verify_batched(x, idx, wt, layer, pool, get_host), {"C": 1, "max_tokens_one_expert": s, "used": int(idx.unique().numel())}
        if C is None:                                  # bucket rows grow with the pass so the bucket COUNT (and the
            C = 64                                     # concatenated weight, 1.57 MB per bucket in bfp8) stays ~600
            while C * 64 < s: C *= 2
        counts = torch.bincount(idx.flatten(), minlength=E)
        used = torch.nonzero(counts).flatten().tolist()
        slots = dict(zip(used, pool.ensure([(layer, e) for e in used], get_host)))
        return self._batched(x, idx, wt, counts, C, lambda e: slots[e]), {"C": C, "max_tokens_one_expert": int(counts.max()), "used": len(used)}

    def verify_batched(self, x, idx, wt, layer, pool, get_host, prof=None, gu_sh=None):
        """Multi-token continuation (DSpark verify, s <= 32): the same batched dispatch over the pool's LRU
        (`ensure`, so the decode's resident set is kept warm rather than scratched), one bucket per expert."""
        import time
        s = x.shape[2]
        if s <= 8 and self.slot_cat is not None:
            idx_l = idx.tolist()
            used = sorted({e for row in idx_l for e in row})
            if prof is not None: ttnn.synchronize_device(self.dev); t0 = time.perf_counter(); m0 = pool.misses
            used_slots = pool.ensure([(layer, e) for e in used], get_host)
            if prof is not None: ttnn.synchronize_device(self.dev); prof["moe ensure"] += time.perf_counter() - t0; prof["misses"] += pool.misses - m0; prof["unique"] += len(used)
            n_u = len(used)
            n_pad = -(-n_u // TOPK) * TOPK
            wt_pad = torch.zeros(n_pad, s, dtype=torch.bfloat16)
            e_to_u = {e: i for i, e in enumerate(used)}
            wt_bf16 = wt.to(torch.bfloat16)
            for r in range(s):
                wt_pad[[e_to_u[e] for e in idx_l[r]], r] = wt_bf16[r]
            slot_list = used_slots + [used_slots[-1]] * (n_pad - n_u)
            acc = None
            for g0 in range(0, n_pad, TOPK):
                self.slot_cat.update_ctrl_multi(slot_list[g0 : g0 + TOPK], wt_pad[g0 : g0 + TOPK])
                self.slot_cat.run_from_ctrl(self.fmt, self.cat)
                g = mm(x, self.cat["w1"])
                u = mm(x, self.cat["w3"])
                if g0 == 0:
                    if gu_sh is None:
                        gu_sh = mm(x, self.sh_w13)
                    h_routed, h_sh = self.slot_cat.swiglu_from_ctrl(g, u, gu_sh, s_rows=s)
                    acc = ttnn.add(mm(h_sh, self.w["sh_w2"]), mm(h_routed, self.cat["w2"]))
                else:
                    h_routed, _ = self.slot_cat.swiglu_from_ctrl(g, u, None, s_rows=s)
                    acc = ttnn.add(acc, mm(h_routed, self.cat["w2"]))
            return all_reduce_sum(acc)
        counts = torch.bincount(idx.flatten(), minlength=NEXP)
        used = torch.nonzero(counts).flatten().tolist()
        if prof is not None: ttnn.synchronize_device(self.dev); t0 = time.perf_counter(); m0 = pool.misses
        slots = dict(zip(used, pool.ensure([(layer, e) for e in used], get_host)))
        if prof is not None: ttnn.synchronize_device(self.dev); prof["moe ensure"] += time.perf_counter() - t0; prof["misses"] += pool.misses - m0; prof["unique"] += len(used)
        return self._batched_small(x, idx, wt, used, lambda e: slots[e])

    def _batched_small(self, x, idx, wt, used, slot_of, C=32, NB=32):
        """s <= 32 tokens: one bucket of C rows per used expert (FIXED NB buckets so every program shape is constant:
        a new shape after the decode traces are captured is the trace hazard). Dispatch and weighted combine are two
        exact one-hot matmuls (ttnn.embedding is slow for this), the expert math is three batched matmuls."""
        s, d = x.shape[2], x.shape[3]; assert len(used) <= NB, len(used)
        first = {e: b for b, e in enumerate(used)}
        owner = [slot_of(e) for e in used] + [slot_of(used[-1])] * (NB - len(used))
        D = torch.zeros(NB * C, 32); W = torch.zeros(32, NB * C)                              # rows of x are tile-padded to 32
        fill = {e: 0 for e in used}
        for t in range(s):
            for k in range(TOPK):
                e = int(idx[t, k]); r = first[e] * C + fill[e]; fill[e] += 1
                D[r, t] = 1.0; W[t, r] = float(wt[t, k])
        xe = ttnn.reshape(mm(to_dev(self.dev, D.view(1, 1, NB * C, 32)), x if s == 32 else ttnn.concat([x, to_dev(self.dev, torch.zeros(1, 1, 32 - s, d))], dim=2)), (1, NB, C, d))
        cat = lambda pj: ttnn.concat([sl[pj] for sl in owner], dim=1)
        g = ttnn.clamp(mm(xe, cat("w1")), max=LIMIT); u = ttnn.clamp(mm(xe, cat("w3")), min=-LIMIT, max=LIMIT)
        y = mm(ttnn.mul(ttnn.silu(g), u), cat("w2"))                                             # [1,NB,C,5120] partial
        routed = mm(to_dev(self.dev, W.view(1, 1, 32, NB * C)), ttnn.reshape(y, (1, 1, NB * C, d)))[:, :, :s, :]
        return all_reduce_sum(ttnn.add(self.shared(x), routed))

    def _batched(self, x, idx, wt, counts, C, slot_of):
        s, d, E = x.shape[2], x.shape[3], NEXP
        nb = (counts + C - 1) // C                                                          # virtual buckets per expert
        first = torch.cumsum(nb, 0) - nb                                                    # first bucket of expert e
        NB = int(-(-int(nb.sum()) // 32) * 32)
        owner = torch.repeat_interleave(torch.arange(E), nb).tolist(); owner += [owner[-1]] * (NB - len(owner))
        owner = [slot_of(e) for e in owner]
        order = torch.argsort(idx.flatten(), stable=True)                                   # picks sorted by expert
        exp_of = idx.flatten()[order]; rank = torch.arange(len(order)) - torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts)
        row = first[exp_of] * C + rank                                                      # bucket row of every pick
        bucket = torch.zeros(NB * C, dtype=torch.int32); bucket[row] = (order // TOPK).to(torch.int32)
        back = torch.zeros(s * TOPK, dtype=torch.int32); back[order] = row.to(torch.int32)
        u32 = lambda t: ttnn.from_torch(t.view(1, -1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.dev, mesh_mapper=rep(self.dev))
        xe = ttnn.reshape(ttnn.embedding(u32(bucket), ttnn.reshape(x, (s, d))), (1, NB, C, d))
        cat = lambda pj: ttnn.concat([sl[pj] for sl in owner], dim=1)
        # one batched weight alive at a time: at s=8192 each concat is ~1 GB and two of them were the OOM
        W = cat("w1"); g = ttnn.clamp(mm(xe, W), max=LIMIT); ttnn.deallocate(W)
        W = cat("w3"); u = ttnn.clamp(mm(xe, W), min=-LIMIT, max=LIMIT); ttnn.deallocate(W); ttnn.deallocate(xe)
        W2 = cat("w2"); y = mm(ttnn.mul(ttnn.silu(g), u), W2)                                # [1,NB,C,5120] partial
        for t in (W2, g, u): ttnn.deallocate(t)
        # gather back as [1, 6, s, d] (pick k before token), NOT [1, s, 6, d]: a trailing-but-one dim of 6 is tile-padded
        # to 32, which made this one tensor 1.34 GB at s=4096 and was the prefill OOM.
        back_kt = back.view(s, TOPK).T.contiguous().flatten()
        y6 = ttnn.reshape(ttnn.embedding(u32(back_kt), ttnn.reshape(y, (NB * C, d))), (1, TOPK, s, d)); ttnn.deallocate(y)
        routed = ttnn.sum(ttnn.mul(y6, to_dev(self.dev, wt.float().T.contiguous().view(1, TOPK, s, 1))), dim=1, keepdim=True); ttnn.deallocate(y6)
        return all_reduce_sum(ttnn.add(self.shared(x), routed))

    def prefill(self, x, idx, wt, get_expert):
        """x [1,1,s,5120]. One-hot dispatch matrices do gather and scatter-add as exact matmuls (v0)."""
        s = x.shape[2]
        acc = self.shared(x)
        for e in idx.unique().tolist():
            rows, cols = torch.where(idx == e)
            n = len(rows); npad = -(-n // 32) * 32
            D = torch.zeros(npad, s); D[torch.arange(n), rows] = 1.0
            Dw = torch.zeros(s, npad); Dw[rows, torch.arange(n)] = wt[rows, cols]          # weights folded into the combine
            self.stream(0, get_expert(e))
            xe = mm(to_dev(self.dev, D.view(1, 1, npad, s)), x)                           # gather
            ye = self._ffn(xe, self.slots[0])
            acc = ttnn.add(acc, mm(to_dev(self.dev, Dw.view(1, 1, s, npad)), ye))         # weighted scatter-add
        return all_reduce_sum(acc)
