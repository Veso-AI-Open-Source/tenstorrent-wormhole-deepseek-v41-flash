"""Attention on the mesh, v0: stock ops, bf16, TP-8 (8 of 64 query heads per chip = one output group
per chip, so the grouped low-rank output projection needs no intra-op communication).

K and V are the SAME single 512-d latent (one KV head). Softmax runs over the visible keys plus one
learned sink logit per head whose value is zero. The output is un-rotated with the query's own
rotation before the output projection. Reference: model.py Attention.forward + kernel.py sparse_attn.
"""
import torch
import ttnn
from mesh import to_dev, shard, TP, mm, rms_norm, blend_row, rep
import rope

H, D, WIN = 64, 512, 128
NEG = -1.0e9
import os as _os
PAD_SOFTMAX = _os.environ.get("DSV41_PAD_SOFTMAX", "1") != "0"


def all_reduce_sum(x):
    """Row-parallel combine. all_gather along dim 1 then sum: the CCL idiom proven on this box's ring."""
    return ttnn.sum(ttnn.all_gather(x, dim=1), dim=1, keepdim=True)


class StepCtx:
    """Everything about a decode step that depends only on the POSITION, uploaded ONCE per token and
    shared by all 40 layers (the profile showed ~8 tiny host uploads per layer, ~3 ms of every layer).
    Two lives: `StepCtx(dev, pos, ...)` allocates fresh tensors (eager decode); `ctx.update(pos)` rewrites
    the SAME device buffers in place, which is what a captured decode trace reads (s18: buffer addresses
    are baked into the trace, so per-token data must land at fixed addresses)."""
    def __init__(self, dev, pos, ang_win, ang_csa, nmax2, nmax1):
        self.dev, self.ang_win, self.ang_csa, self.nmax2, self.nmax1 = dev, ang_win, ang_csa, nmax2, nmax1
        self.bufs = None
        self.update(pos)

    def _host(self, pos):
        """The step's tensors on the host, in a fixed order (same order every time: names -> torch)."""
        def onehot(n, j):
            hot = torch.zeros(1, 1, n, 1)
            if j is not None and j < n: hot[0, 0, j, 0] = 1.0
            return hot, 1.0 - hot
        tab = lambda ang: [t.view(1, 1, 1, -1) for t in (ang.cos().repeat_interleave(2, -1), ang.sin().repeat_interleave(2, -1))]
        g2 = max(pos - 1, 0)
        h = {}
        h["cos_win"], h["sin_win"] = tab(self.ang_win[pos])
        h["cos_csa"], h["sin_csa"] = tab(self.ang_csa[pos])
        h["cos_grp2"], h["sin_grp2"] = tab(self.ang_csa[g2])                                     # a ratio-2 group sits at its first token
        h["ring_hot"], h["ring_keep"] = onehot(WIN, pos % WIN)
        h["ring_mask"] = torch.where(torch.arange(WIN) <= pos, 0.0, NEG).view(1, 1, 1, WIN)
        h["c2_hot"], h["c2_keep"] = onehot(self.nmax2, pos // 2 if (pos + 1) % 2 == 0 else None)
        h["c1_hot"], h["c1_keep"] = onehot(self.nmax1, pos)
        h["reach2"] = torch.where(torch.arange(self.nmax2) < (pos + 1) // 2, 0.0, NEG).view(1, 1, 1, self.nmax2)
        h["reach1"] = torch.where(torch.arange(self.nmax1) < pos + 1, 0.0, NEG).view(1, 1, 1, self.nmax1)
        pos_u16 = torch.zeros(1, 1, 1, 32, dtype=torch.uint16)
        pos_u16[0, 0, 0, 0] = 0x4000 | (pos % WIN)
        pos_u16[0, 0, 0, 1] = (0x4000 | (pos // 2)) if (pos + 1) % 2 == 0 and (pos // 2) < self.nmax2 else 0x7FFF
        pos_u16[0, 0, 0, 2] = (0x4000 | pos) if pos < self.nmax1 else 0x7FFF
        pos_u16[0, 0, 0, 3] = 0x4000 | min(pos + 1, WIN)
        pos_u16[0, 0, 0, 4] = 0x4000 | min((pos + 1) // 2, self.nmax2)
        pos_u16[0, 0, 0, 5] = 0x4000 | min(pos + 1, self.nmax1)
        h["pos_buf"] = pos_u16.view(torch.bfloat16)
        return h

    def update(self, pos):
        prev_pos = getattr(self, "pos", None)
        self.pos = pos; h = self._host(pos)
        if self.bufs is None:
            self.bufs = {
                k: (
                    ttnn.from_torch(v.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.dev, mesh_mapper=rep(self.dev))
                    if k == "pos_buf"
                    else to_dev(self.dev, v)
                )
                for k, v in h.items()
            }
        else:
            skip = {"ring_hot", "ring_keep", "c2_hot", "c2_keep", "c1_hot", "c1_keep"}
            if (pos + 1) <= 512:
                skip.update(("ring_mask", "reach2", "reach1"))
            if prev_pos == pos - 1:
                if pos > WIN:
                    skip.add("ring_mask")
                if pos % 2 == 0:
                    skip.update(("cos_grp2", "sin_grp2", "reach2"))
            for k, v in h.items():
                if k in skip:
                    continue
                ttnn.copy_host_to_device_tensor(ttnn.from_torch(v.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep(self.dev)), self.bufs[k])
        b = self.bufs
        self.cs_win, self.cs_csa, self.cs_grp2 = (b["cos_win"], b["sin_win"]), (b["cos_csa"], b["sin_csa"]), (b["cos_grp2"], b["sin_grp2"])
        self.ring_hot, self.ring_keep, self.ring_mask = b["ring_hot"], b["ring_keep"], b["ring_mask"]
        self.c2_hot, self.c2_keep, self.c1_hot, self.c1_keep = b["c2_hot"], b["c2_keep"], b["c1_hot"], b["c1_keep"]
        self.reach = {2: b["reach2"], 1: b["reach1"]}
        self.pos_buf = b["pos_buf"]
        return self

    @staticmethod
    def blend(cache, hot, keep, row):
        return ttnn.add(ttnn.mul(cache, keep), mm(hot, row))

    @staticmethod
    def blend_into(cache, hot, keep, row):
        """cache[row j] = row IN PLACE (the cache keeps its address, so a trace can carry it across tokens)."""
        ttnn.copy(StepCtx.blend(cache, hot, keep, row), cache)
        return cache


class MultiCtx:
    """Position data for a multi-token continuation of s tokens at pos..pos+s-1 (DSpark verify), uploaded once per
    pass and shared by all layers: rope tables (window / CSA / ratio-2 group starts), the ring one-hot [128,s], the
    window mask, and per-ratio reach masks for the compressed caches."""
    def __init__(self, dev, pos, s, ang_win, ang_csa, nmax2, nmax1):
        self.dev, self.pos, self.s, self.ang_csa, self.nmax = dev, pos, s, ang_csa, {2: nmax2, 1: nmax1}
        self.cs_win = rope.tables(dev, ang_win[pos : pos + s])
        self.cs_csa = rope.tables(dev, ang_csa[pos : pos + s])
        self._cache = {}

    def _ensure_ring_P(self):
        if "ring_P" not in self._cache:
            P = torch.zeros(WIN, self.s)
            for i in range(self.s): P[(self.pos + i) % WIN, i] = 1.0
            self._cache["ring_P"] = to_dev(self.dev, P.view(1, 1, WIN, self.s))
            self._cache["ring_keep"] = to_dev(self.dev, (1.0 - P.sum(1)).view(1, 1, WIN, 1))

    @property
    def ring_P(self):
        self._ensure_ring_P()
        return self._cache["ring_P"]

    @property
    def ring_keep(self):
        self._ensure_ring_P()
        return self._cache["ring_keep"]

    def window_mask(self, s):
        """[1,1,s,127+s]: query pos+i sees keys with absolute position in (pos+i-128, pos+i] AND >= 0 (a ring
        tail before position 0 holds the prefill's zero padding, which the single-token path masks via ring_mask)."""
        if ("wm", s) not in self._cache:
            qi = (torch.arange(s) + self.pos)[:, None]; ki = (torch.arange(-(WIN - 1), s) + self.pos)[None, :]
            ok = (ki <= qi) & (ki > qi - WIN) & (ki >= 0)
            self._cache[("wm", s)] = to_dev(self.dev, torch.where(ok, 0.0, NEG).view(1, 1, s, WIN - 1 + s))
        return self._cache[("wm", s)]

    def window_mask_ring(self, s):
        """[1,1,s,WIN+s]: causal window mask over tile-aligned [self.win (128), kv (s)]."""
        if ("wmr", s) not in self._cache:
            qi = (torch.arange(s) + self.pos)[:, None]
            j = torch.arange(WIN)
            p_ring = torch.where(j < self.pos, (self.pos - 1) - ((self.pos - 1 - j) % WIN), -1)[None, :]
            ok_ring = (p_ring >= 0) & (p_ring <= qi) & (p_ring > qi - WIN)
            ok_new = torch.arange(s)[None, :] <= torch.arange(s)[:, None]
            ok = torch.cat([ok_ring, ok_new], dim=1)
            self._cache[("wmr", s)] = to_dev(self.dev, torch.where(ok, 0.0, NEG).view(1, 1, s, WIN + s))
        return self._cache[("wmr", s)]

    def csa_short_mask(self, r, ckv_w, s):
        """[1,1,s,WIN+ckv_w+s]: combined mask over tile-aligned [self.win (128), ckv_active (ckv_w), kv_w (s)]."""
        key = ("csa_sm", r, ckv_w, s)
        if key not in self._cache:
            qi = (torch.arange(s) + self.pos)[:, None]
            j = torch.arange(WIN)
            p_ring = torch.where(j < self.pos, (self.pos - 1) - ((self.pos - 1 - j) % WIN), -1)[None, :]
            ok_ring = (p_ring >= 0) & (p_ring <= qi) & (p_ring > qi - WIN)
            ok_ckv = torch.arange(ckv_w)[None, :] < ((qi + 1) // r)
            ok_new = torch.arange(s)[None, :] <= torch.arange(s)[:, None]
            ok = torch.cat([ok_ring, ok_ckv, ok_new], dim=1)
            self._cache[key] = to_dev(self.dev, torch.where(ok, 0.0, NEG).view(1, 1, s, WIN + ckv_w + s))
        return self._cache[key]

    def reach(self, r, width=None):
        """[1,1,s,width] additive: entry j visible to query t = pos+i iff j < (t+1)//r."""
        w = self.nmax[r] if width is None else width
        if ("reach", r, w) not in self._cache:
            t = torch.arange(self.pos, self.pos + self.s)[:, None]; j = torch.arange(w)[None, :]
            self._cache[("reach", r, w)] = to_dev(self.dev, torch.where(j < (t + 1) // r, 0.0, NEG).view(1, 1, self.s, w))
        return self._cache[("reach", r, w)]

    def cache_P(self, r, n_prev, n_new):
        """One-hot [nmax, n_new] placing new compressed entries at rows n_prev.., and its keep column."""
        P = torch.zeros(self.nmax[r], n_new)
        for i in range(n_new): P[n_prev + i, i] = 1.0
        return to_dev(self.dev, P.view(1, 1, self.nmax[r], n_new)), to_dev(self.dev, (1.0 - P.sum(1)).view(1, 1, self.nmax[r], 1))

    def cs_groups(self, g0, n_new, r):
        if r == 1 and g0 == self.pos and n_new == self.s:
            return self.cs_csa
        key = ("cs_g", g0, n_new, r)
        if key not in self._cache:
            self._cache[key] = rope.tables(self.dev, self.ang_csa[g0 : g0 + n_new * r : r])
        return self._cache[key]


class Attention:
    def __init__(self, dev, w, rope_base=10000.0, yarn=False, max_pos=4096):
        self.dev, self.w = dev, w
        self.scale = D ** -0.5
        self.angles = rope.freqs(max_pos, rope_base, 65536 if yarn else 0)
        self.sink = w["attn_sink"].float()                                   # [64], host; uploaded per shape
        self.win = to_dev(dev, torch.zeros(1, 1, WIN, D))                    # decode ring buffer, replicated
        self.sink1 = self._sink_col(1)                                       # decode sink column, uploaded once
        self._pad1 = {}                                                      # decode softmax pad columns by width

    # ---- shared pieces
    def _q(self, x, cos, sin):
        s = x.shape[2]
        qr = rms_norm(mm(x, self.w["wq_a"]), self.w["q_norm"])
        q = mm(qr, self.w["wq_b"])                                   # [1,1,s,8*512] per chip
        if s <= 32:
            from rope_cache_op import rope_reshape_q1
            return qr, rope_reshape_q1(self.dev, q, cos, sin)
        q = ttnn.transpose(ttnn.reshape(q, (1, s, H // TP, D)), 1, 2)         # [1,8,s,512]
        return qr, rope.apply(q, cos, sin)

    def _kv(self, x, cos, sin, apply_rope=True):
        kv = rms_norm(mm(x, self.w["wkv"]), self.w["kv_norm"])
        return rope.apply(kv, cos, sin) if apply_rope else kv                 # [1,1,s,512]

    def _sink_col(self, s):
        col = self.sink.view(1, H, 1, 1).expand(1, H, s, 1).contiguous()
        return to_dev(self.dev, col, shard(self.dev, 1))                      # [1,8,s,1] per chip

    def _attend(self, q, keys, mask, cos, sin, pos_buf=None, mode_slot=0):
        """q [1,8,s,512], keys [1,1,k,512], mask [1,1,s,k] additive or None."""
        s = q.shape[2]
        if s == 1 and pos_buf is not None:
            from rope_cache_op import attn_softmax_1, rope_reshape_o1
            sc = mm(q, keys, transpose_b=True)
            p = attn_softmax_1(sc, self.sink1, pos_buf, mode_slot)
            o = rope_reshape_o1(self.dev, mm(p, keys), cos, sin)
            return all_reduce_sum(mm(mm(o, self.w["wo_a"]), self.w["wo_b"]))
        sc = ttnn.mul(mm(q, keys, transpose_b=True), self.scale)                 # [1,8,s,k]
        if mask is not None:
            sc = ttnn.add(sc, mask)
        k = sc.shape[3]
        pad = (-(k + 1)) % 256
        if s <= 32 and pad and PAD_SOFTMAX:
            key = pad if s == 1 else (s, pad)
            if key not in self._pad1:
                sink_s = self.sink1 if s == 1 else self._sink_col(s)
                self._pad1[key] = ttnn.concat([sink_s, to_dev(self.dev, torch.full((1, H, s, pad), NEG), shard(self.dev, 1))], dim=3)
            cols = [sc, self._pad1[key]]
        else:
            cols = [sc, self.sink1 if s == 1 else self._sink_col(s)]
            if pad and PAD_SOFTMAX:
                cols.append(to_dev(self.dev, torch.full((1, H, s, pad), NEG), shard(self.dev, 1)))
        p = ttnn.softmax(ttnn.concat(cols, dim=3), dim=-1)[:, :, :, :k]
        if s <= 32:
            from rope_cache_op import rope_reshape_o1
            o = rope_reshape_o1(self.dev, mm(p, keys), cos, sin)
        else:
            o = rope.apply(mm(p, keys), cos, sin, inverse=True)                      # [1,8,s,512]
            o = ttnn.reshape(ttnn.transpose(o, 1, 2), (1, 1, s, (H // TP) * D))      # this chip's group
        return all_reduce_sum(mm(mm(o, self.w["wo_a"]), self.w["wo_b"]))

    # ---- window-only layers (compress_ratio 0)
    def ring_tail(self, start_pos):
        """The previous pass's last 127 window keys in POSITION order (the ring stores slot = pos % 128)."""
        k = start_pos % WIN
        ordered = self.win if k == 0 else ttnn.concat([self.win[:, :, k:, :], self.win[:, :, :k, :]], dim=2)
        return ordered[:, :, 1:, :]                                        # drop the oldest: positions start-127 .. start-1

    @staticmethod
    def window_mask(dev, cache, q0, q1, k0, start_pos):
        """Additive [1,1,q1-q0,q1-k0] causal-window mask for LOCAL query rows [q0,q1) over LOCAL key rows [k0,q1).
        With a continuation, local key row j < 0 means the ring tail; positions are relative so one mask serves all passes."""
        key = ("win", q0, q1, k0)
        if key not in cache:
            qi, ki = torch.arange(q0, q1)[:, None], torch.arange(k0, q1)[None, :]
            cache[key] = to_dev(dev, torch.where((ki <= qi) & (ki > qi - WIN), 0.0, NEG).view(1, 1, q1 - q0, q1 - k0))
        return cache[key]

    def prefill_window(self, x, start_pos=0, chunk=1024, cache=None):
        """x [1,1,s,5120] (post-norm). Causal sliding window of 128, queries in chunks; seeds the ring.
        start_pos > 0 continues a sequence: the first 127 keys come from the ring (pass sizes are multiples of 128)."""
        s = x.shape[2]; cache = {} if cache is None else cache
        cos, sin = rope.tables(self.dev, self.angles[start_pos : start_pos + s])
        _, q = self._q(x, cos, sin); kv = self._kv(x, cos, sin); outs = []
        T = WIN - 1 if start_pos else 0
        keys_all = ttnn.concat([self.ring_tail(start_pos), kv], dim=2) if start_pos else kv          # local key row j lives at index j + T
        for q0 in range(0, s, chunk):
            q1 = min(s, q0 + chunk); k0 = max(-T, q0 - WIN + 1); n = q1 - q0
            tab = (cos, sin) if n == s else (cos[:, :, q0:q1, :], sin[:, :, q0:q1, :])
            ks = keys_all if (k0 + T == 0 and q1 + T == keys_all.shape[2]) else keys_all[:, :, k0 + T : q1 + T, :]
            outs.append(self._attend(q if n == s else q[:, :, q0:q1, :], ks, self.window_mask(self.dev, cache, q0, q1, k0, start_pos), *tab))
        self.ring_from_prefill(kv, s, start_pos)
        return outs[0] if len(outs) == 1 else ttnn.concat(outs, dim=2)

    def ring_from_prefill(self, kv, s, start_pos=0):
        """Seed the decode ring from a pass's window keys kv [1,1,s,512] at positions start_pos..: slot j = the last
        position p with p % 128 == j. A short first pass (s < 128 at position 0) leaves the tail zero; a short
        continuation blends into the ring it continues."""
        if s >= WIN:
            e = (start_pos + s) % WIN; i0 = s - e if e else s - WIN
            self.win = kv[:, :, s - WIN :, :] if e == 0 else ttnn.concat([kv[:, :, i0:s, :], kv[:, :, s - WIN : i0, :]], dim=2)
        elif start_pos == 0:
            self.win = ttnn.concat([kv, to_dev(self.dev, torch.zeros(1, 1, WIN - s, D))], dim=2)
        else:
            P = torch.zeros(WIN, s)
            for i in range(s): P[(start_pos + i) % WIN, i] = 1.0
            self.win = ttnn.add(ttnn.mul(self.win, to_dev(self.dev, (1.0 - P.sum(1)).view(1, 1, WIN, 1))), mm(to_dev(self.dev, P.view(1, 1, WIN, s)), kv))

    def ring_mask(self, pos):
        return None if pos >= WIN - 1 else to_dev(self.dev, torch.where(torch.arange(WIN) <= pos, 0.0, NEG).view(1, 1, 1, WIN))

    def load_ring(self, ring: torch.Tensor):
        """Seed the decode ring [128,512] (slot j holds the key of the last position p with p % 128 == j)."""
        self.win = to_dev(self.dev, ring.view(1, 1, WIN, D))

    # ---- multi-token continuation at an arbitrary position (DSpark verify): s <= 32 tokens through the DECODE
    # semantics (full history), every piece of state updated in place so captured decode traces stay valid.
    def window_keys_multi(self, kv, pos, ctxm):
        """Keys/mask for s new window keys kv [1,1,s,512] at positions pos..pos+s-1: the ring's last 127 keys in
        position order + the new keys; query i sees local key rows i..i+127."""
        s = kv.shape[2]
        return ttnn.concat([self.ring_tail(pos), kv], dim=2), ctxm.window_mask(s)

    def ring_write_multi(self, kv, ctxm):
        """ring[(pos+i) % 128] = kv[i] for all i, in place (ctxm.ring_P [128,s] one-hot, ring_keep [128,1])."""
        if kv.shape[2] <= 32:
            from rope_cache_op import cache_write_multi
            cache_write_multi(self.win, kv, ctxm.pos, kv.shape[2], is_ring=True)
            return
        ttnn.copy(ttnn.add(ttnn.mul(self.win, ctxm.ring_keep), mm(ctxm.ring_P, kv)), self.win)

    def decode_multi(self, x, pos, ctxm):
        cos, sin = ctxm.cs_win
        _, q = self._q(x, cos, sin); kv = self._kv(x, cos, sin)
        keys = ttnn.concat([self.win, kv], dim=2)
        out = self._attend(q, keys, ctxm.window_mask_ring(kv.shape[2]), cos, sin)
        self.ring_write_multi(kv, ctxm)
        return out

    def decode_window(self, x, pos, ctx=None):
        """x [1,1,1,5120] at absolute position `pos`: write this key into slot pos % 128, attend over the ring."""
        if ctx is None:
            ctx = StepCtx(self.dev, pos, self.angles, self.angles, 512, 512)
        cos, sin = ctx.cs_win
        _, q = self._q(x, cos, sin)
        from rope_cache_op import cache_write_1
        cache_write_1(self.win, self._kv(x, cos, sin, apply_rope=False), ctx.pos_buf, 0, cos=cos, sin=sin)
        return self._attend(q, self.win, ctx.ring_mask, cos, sin, pos_buf=ctx.pos_buf, mode_slot=0)
