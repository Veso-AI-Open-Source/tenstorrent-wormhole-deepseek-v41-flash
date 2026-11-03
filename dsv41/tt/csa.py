"""Compressed Sparse Attention 2 on the mesh, v0 PREFILL (stock ops).

Three layer modes (model.py Attention/Compressor/Indexer):
  full    : this layer compresses its own KV (one 512-d latent per `ratio` tokens), owns the index
            keys, and runs the indexer. Layers 2, 8, 14 (ratio 2) and 20 (ratio 1, candidate source).
  reindex : reuses the source's compressed KV and index keys, runs its OWN index query. 24, 28, 32, 36.
  reuse   : reuses compressed KV AND the selection published by the last index source.

v0 trick: the reference gathers each query's top-512 compressed entries and attends over them. Dense
attention over ALL compressed entries with an additive mask that is 0 on exactly those entries is the
same function, needs no gather/scatter, and costs O(s * s/ratio): right for bring-up (<= a few K
tokens), wrong for long context.
"""
import torch
import ttnn
from mesh import to_dev, mm, TP, rms_norm, blend_row
from attn import Attention, StepCtx, all_reduce_sum, NEG, WIN, H, D
import rope

IDX_HEADS, IDX_DIM, TOPK = 32, 128, 512
import os as _os
# env overrides exist so the long-context code paths can be exercised (and gated) on short sequences
BUCKET = int(_os.environ.get("DSV41_BUCKET", 16384)); SCORE_BYTES = int(_os.environ.get("DSV41_SCORE_MB", 256)) << 20; COMPACT_BYTES = int(_os.environ.get("DSV41_COMPACT_MB", 48)) << 20


class Shared:
    """What a source layer publishes for the layers after it (model.py SharedAttentionRuntime)."""
    compress_kv = None   # [1,1,n_c,512] rotated
    index_k = None       # [1,1,n_c,128] rotated
    mask_c = None        # [1,1,s,n_c] additive selection mask (prefill, single chunk)
    masks = None         # prefill: per-query-chunk selection masks published by the last index source
    ratio = 0
    ckv = None           # decode: the source's persistent compressed-KV cache [1,1,Nmax,512]
    ik = None            # decode: the source's persistent index-key cache     [1,1,Nmax,128]
    mask_dec = None      # decode: [1,1,1,Nmax] additive selection mask (one fixed buffer per ratio, see mask_bufs)
    mask_multi = None    # verify pass: [1,1,s,Nmax] selection mask published by the last index source
    uniform = False      # decode: take the branch-free path (a trace must not depend on the position)

    def __init__(self):
        self.mask_bufs = {}


class CSA(Attention):
    def __init__(self, dev, w, ratio, mode, max_pos=4096):
        super().__init__(dev, w, rope_base=160000.0, yarn=True, max_pos=max_pos)
        self.ratio, self.mode = ratio, mode
        self.nmax, self.st = max_pos // ratio, None

    # ---- compressor: [1,1,s,5120] -> latent [1,1,s/ratio,512], normed, NOT yet rotated
    def _pool(self, kv, sc, n):
        """softmax-gated pooling of `ratio` consecutive tokens, per channel. kv/sc [1,1,n*r,512] -> [1,1,n,512]."""
        pair = lambda t: ttnn.transpose(ttnn.reshape(t, (1, n, self.ratio, D)), 2, 3)                # [1,n,512,r]
        return ttnn.reshape(ttnn.sum(ttnn.mul(pair(kv), ttnn.softmax(pair(sc), dim=-1)), dim=3, keepdim=True), (1, 1, n, D))

    def compress(self, x):
        """[1,1,s,5120] -> latent [1,1,s//ratio,512], normed, NOT yet rotated. A trailing partial group
        (ratio 2, odd s) is kept as decode state, exactly like the reference's kv_state/score_state."""
        s = x.shape[2]; n = s // self.ratio
        kv = mm(x, self.w["comp_wkv"])
        if self.ratio > 1:
            sc = mm(x, self.w["comp_wgate"])
            self.st = (kv[:, :, s - 1 : s, :], sc[:, :, s - 1 : s, :]) if s % self.ratio else None
            kv = self._pool(kv[:, :, : n * self.ratio, :], sc[:, :, : n * self.ratio, :], n)
        return rms_norm(kv, self.w["comp_norm"])

    def index_keys(self, latent, cos_c, sin_c):
        k = rms_norm(mm(latent, self.w["idx_wk"]), self.w["idx_k_norm"])
        return rope.apply(k, cos_c, sin_c)                                                   # [1,1,n_c,128]

    def index_scores(self, x, qr, index_k, cos, sin):
        """sum_h relu(q_h . k) * w_h  -> [1,1,s,n_c], identical on every chip after the all-reduce."""
        s = x.shape[2]; hl = IDX_HEADS // TP
        q = mm(qr, self.w["idx_wq_b"])
        q = ttnn.reshape(q, (1, hl, 1, IDX_DIM)) if s == 1 else ttnn.transpose(ttnn.reshape(q, (1, s, hl, IDX_DIM)), 1, 2)   # [1,4,s,128]
        q = rope.apply(q, cos, sin)
        wt = ttnn.mul(mm(x, self.w["idx_weights_proj"]), IDX_DIM ** -0.5 * IDX_HEADS ** -0.5)    # [1,1,s,4]
        wt = ttnn.reshape(wt, (1, hl, 1, 1)) if s == 1 else ttnn.transpose(ttnn.reshape(wt, (1, s, hl, 1)), 1, 2)                                 # [1,4,s,1]
        sc = ttnn.mul(ttnn.relu(mm(q, index_k, transpose_b=True)), wt)                         # [1,4,s,n_c]
        return all_reduce_sum(ttnn.sum(sc, dim=1, keepdim=True))

    def select(self, scores, s, n_c, start_pos=0):
        """Additive mask [1,1,s,n_c]: 0 on each query's top-512 REACHABLE entries, NEG elsewhere."""
        t = torch.arange(start_pos, start_pos + s)
        reach = torch.arange(n_c)[None, :] < ((t + 1) // self.ratio)[:, None]
        reach_add = to_dev(self.dev, torch.where(reach, 0.0, NEG).view(1, 1, s, n_c))
        if n_c <= TOPK:
            return reach_add
        masked = ttnn.add(scores, reach_add)
        kth = ttnn.topk(masked, k=TOPK, dim=-1, largest=True, sorted=True)[0][:, :, :, TOPK - 1 : TOPK]
        keep = ttnn.ge(masked, kth)
        return ttnn.add(ttnn.mul(ttnn.sub(keep, 1.0), -NEG), reach_add)                     # keep ? 0 : NEG, then reach

    # ---- prefill, queries processed in chunks so nothing is O(s^2) in memory; start_pos > 0 continues a sequence
    def prefill(self, x, shared: Shared, chunk=1024, start_pos=0, kv_only=False, kv_ready=False, history=True):
        """Window keys for a query chunk are [q0-127, q1) (the first 127 of a continuation come from the ring);
        compressed keys are ALL entries so far with the device-computed top-512 mask. Host-built masks depend only
        on positions, so they are built once per pass and shared by every layer through `shared.cache`.
        Pass sizes are multiples of 128 (ring alignment, ratio-2 pairs never straddle a pass)."""
        s = x.shape[2]; r = self.ratio; n_prev = start_pos // r; n_new = s // r; n_c = n_prev + n_new
        cache = shared.__dict__.setdefault("cache", {})
        cos, sin = rope.tables(self.dev, self.angles[start_pos : start_pos + s])
        qr, q = self._q(x, cos, sin)
        kv_w = self._kv(x, cos, sin)
        if self.mode == "full" and kv_ready:                        # CED tail: the cache already holds every entry up to start_pos + s
            self._ckv_bufs = {}
            w = min(self.nmax, -(-n_c // BUCKET) * BUCKET) if n_c > BUCKET else n_c
            shared.width, shared.ratio = w, r
            shared.compress_kv = self.ckv if w == self.nmax else self.ckv[:, :, :w, :]
            shared.index_k = self.ik if w == self.nmax else self.ik[:, :, :w, :]
        elif self.mode == "full":
            self._ckv_bufs = {}
            cos_c, sin_c = rope.tables(self.dev, self.angles[start_pos : start_pos + n_new * r : r])
            latent = self.compress(x)
            ik_new, ckv_new = self.index_keys(latent, cos_c, sin_c), rope.apply(latent, cos_c, sin_c)
            # persistent caches: [previous entries | this pass | zero tail], rebuilt per pass (no slice-write op for this)
            def grow(old, new, d):
                parts = ([old[:, :, :n_prev, :]] if n_prev else []) + [new]
                if n_c < self.nmax: parts.append(to_dev(self.dev, torch.zeros(1, 1, self.nmax - n_c, d)))
                return parts[0] if len(parts) == 1 else ttnn.concat(parts, dim=2)
            self.ckv, self.ik = grow(getattr(self, "ckv", None), ckv_new, D), grow(getattr(self, "ik", None), ik_new, IDX_DIM)
            # Attend over a cache WIDTH rounded up to a multiple of 16K (entries beyond n_c are masked by `reach`):
            # every distinct width recompiles every attention program, and at 128K there would be 16 of them.
            shared.width = w = min(self.nmax, -(-n_c // BUCKET) * BUCKET) if n_c > BUCKET else n_c
            shared.compress_kv = self.ckv if w == self.nmax else self.ckv[:, :, :w, :]
            shared.index_k = self.ik if w == self.nmax else self.ik[:, :, :w, :]
            shared.ratio = r
        if kv_only:                                                  # CED: this layer only feeds the decoder's global KV for these tokens
            return None
        if self.mode in ("full", "reindex"):
            shared.masks = []
        elif shared.masks is None and shared.mask_c is not None:     # a gate handed us one full [s, n_c] selection mask
            full = shared.mask_c
            shared.masks = [full if s <= chunk else full[:, :, q0 : min(s, q0 + chunk), :] for q0 in range(0, s, chunk)]
        T = WIN - 1 if (start_pos and history) else 0              # history=False: SWA truncated to this segment (bounded replay)
        keys_w = ttnn.concat([self.ring_tail(start_pos), kv_w], dim=2) if T else kv_w
        w = shared.compress_kv.shape[2]                                   # attended compressed width (>= n_c)
        while chunk > 64 and (H // TP) * chunk * (w + chunk + WIN) * 2 > SCORE_BYTES:
            chunk //= 2                                                   # keep the [8, chunk, keys] score tensor under ~600 MB
        compact = chunk * w * 2 > COMPACT_BYTES                           # big selection masks are kept as bfp4 (0 / -1e9 are exact)
        if self.mode in ("full", "reindex"): shared.chunk = chunk
        elif getattr(shared, "chunk", None): chunk = shared.chunk          # a reuse layer walks its source's chunks
        outs = []
        for ci, q0 in enumerate(range(0, s, chunk)):
            q1 = min(s, q0 + chunk); k0 = max(-T, q0 - WIN + 1); n = q1 - q0
            mask_w = self.window_mask(self.dev, cache, q0, q1, k0, start_pos)
            tab = (cos, sin) if n == s else (cos[:, :, q0:q1, :], sin[:, :, q0:q1, :])
            if self.mode in ("full", "reindex"):
                # Reach mask (entry j visible to query t iff j < (t+1)//r) is BUILT ON DEVICE per chunk from two tiny
                # float32 vectors. Caching dense [n, w] masks per chunk cost 1-2 GB at 40K context (the 128K OOM);
                # float32 because bf16 cannot hold integers above 256 exactly.
                ckey, lkey = ("cols", w), ("lim", r, start_pos, q0, q1)
                if ckey not in cache:
                    cache[ckey] = to_dev(self.dev, torch.arange(w, dtype=torch.float32).view(1, 1, 1, w), dtype=ttnn.float32)
                if lkey not in cache:
                    t = torch.arange(start_pos + q0, start_pos + q1)
                    cache[lkey] = to_dev(self.dev, ((t + 1) // r).float().view(1, 1, n, 1), dtype=ttnn.float32)
                reach = ttnn.mul(ttnn.sub(ttnn.typecast(ttnn.lt(cache[ckey], cache[lkey]), ttnn.bfloat16), 1.0), -NEG)   # 0 / -1e9
                xs, qrs = (x, qr) if n == s else (x[:, :, q0:q1, :], qr[:, :, q0:q1, :])
                sc = self.index_scores(xs, qrs, shared.index_k, *tab)
                if n_c <= TOPK:
                    m = reach
                else:
                    masked = ttnn.add(sc, reach)
                    kth = ttnn.topk(masked, k=TOPK, dim=-1, largest=True, sorted=True)[0][:, :, :, TOPK - 1 : TOPK]
                    m = ttnn.add(ttnn.mul(ttnn.sub(ttnn.ge(masked, kth), 1.0), -NEG), reach)
                    if compact: m = ttnn.typecast(m, ttnn.bfloat4_b)
                shared.masks.append(m)
            qc = q if n == s else q[:, :, q0:q1, :]
            kw = keys_w if (k0 + T == 0 and q1 + T == keys_w.shape[2]) else keys_w[:, :, k0 + T : q1 + T, :]
            mc = shared.masks[ci]
            if mc.dtype != ttnn.bfloat16: mc = ttnn.typecast(mc, ttnn.bfloat16)
            outs.append(self._attend(qc, ttnn.concat([kw, shared.compress_kv], dim=2), ttnn.concat([mask_w, mc], dim=3), *tab))
        if self.mode != "reuse":                                                        # gates read the whole selection; skip the big concat on long passes
            shared.mask_c = shared.masks[0] if len(shared.masks) == 1 else (ttnn.concat(shared.masks, dim=2) if s <= 4096 and not start_pos else None)
        self.ring_from_prefill(kv_w, s, start_pos)
        return outs[0] if len(outs) == 1 else ttnn.concat(outs, dim=2)

    # ---- one decode step at absolute position `pos`
    # ---- decode. Every piece of per-layer state (ring, compressed caches, the ratio-2 pair state, the published
    # selection mask) is updated IN PLACE at a fixed address, so the whole step can be captured as a trace (s18).
    def st_bufs(self):
        if getattr(self, "_st_bufs", None) is None:
            self._st_bufs = (to_dev(self.dev, torch.zeros(1, 1, 1, D)), to_dev(self.dev, torch.zeros(1, 1, 1, D)))
        return self._st_bufs

    def adopt_state(self):
        """Move a prefill's trailing pair state into the fixed pair buffers (call once before decoding)."""
        if self.ratio == 2 and self.st is not None and self.st is not self.st_bufs():
            for src, dst in zip(self.st, self.st_bufs()):
                ttnn.copy(src, dst)
            self.st = self.st_bufs()

    def decode_multi(self, x, pos, shared: Shared, ctxm):
        """s tokens at pos..pos+s-1 through the decode semantics (DSpark verify). Compressed-cache entries are
        written in place at their absolute group index; ratio-2 pairs align to absolute positions (an odd pos
        joins the parked row of pos-1); a trailing odd row is parked. Entries beyond the accepted length after a
        rollback are never reachable before they are rewritten, so only the pair state needs restoring."""
        r, s = self.ratio, x.shape[2]
        self.adopt_state()
        cos, sin = ctxm.cs_csa
        qr, q = self._q(x, cos, sin); kv_w = self._kv(x, cos, sin)
        self.last_rows = None
        if self.mode == "full":
            kv = mm(x, self.w["comp_wkv"])
            if r == 2:
                sc = mm(x, self.w["comp_wgate"]); bk, bs = self.st_bufs()
                if pos % 2:                                        # the open pair (pos-1, pos): parked row first
                    kv, sc = ttnn.concat([bk, kv], dim=2), ttnn.concat([bs, sc], dim=2)
                g0 = pos - pos % 2; n_rows = s + pos % 2; n_new = n_rows // 2
                self.last_rows = (kv, sc, g0)                      # rollback: the parked row of the accepted length
                if n_rows % 2:
                    ttnn.copy(kv[:, :, n_rows - 1 : n_rows, :], bk); ttnn.copy(sc[:, :, n_rows - 1 : n_rows, :], bs); self.st = (bk, bs)
                else:
                    self.st = None
                latent = self._pool(kv[:, :, : 2 * n_new, :], sc[:, :, : 2 * n_new, :], n_new) if n_new else None
                n_prev = g0 // 2
            else:
                latent, n_new, n_prev, g0 = kv, s, pos, pos
            if latent is not None:
                latent = rms_norm(latent, self.w["comp_norm"])
                cos_c, sin_c = ctxm.cs_groups(g0, n_new, r)
                if n_new <= 32:
                    from rope_cache_op import cache_write_multi
                    ik_raw = rms_norm(mm(latent, self.w["idx_wk"]), self.w["idx_k_norm"])
                    cache_write_multi(self.ik, ik_raw, n_prev, n_new, cos=cos_c, sin=sin_c)
                    ckv_rot = rope.apply(latent, cos_c, sin_c)
                    cache_write_multi(self.ckv, ckv_rot, n_prev, n_new)
                    for cw, cb in getattr(self, "_ckv_bufs", {}).items():
                        cache_write_multi(cb, ckv_rot, n_prev, n_new)
                else:
                    P, keep = ctxm.cache_P(r, n_prev, n_new)
                    ttnn.copy(ttnn.add(ttnn.mul(self.ik, keep), mm(P, self.index_keys(latent, cos_c, sin_c))), self.ik)
                    ttnn.copy(ttnn.add(ttnn.mul(self.ckv, keep), mm(P, rope.apply(latent, cos_c, sin_c))), self.ckv)
                    for cw, cb in getattr(self, "_ckv_bufs", {}).items():
                        ttnn.copy(self.ckv if cw == self.nmax else self.ckv[:, :, :cw, :], cb)
            shared.ckv, shared.ik = self.ckv, self.ik
        n_c = (pos + s) // r
        if n_c <= TOPK:
            ckv_w = min(self.nmax, max(64, -(-max(n_c, 1) // 64) * 64))
            if self.mode == "full":
                bufs = self.__dict__.setdefault("_ckv_bufs", {})
                buf = bufs.get(ckv_w)
                if buf is None:
                    buf = bufs[ckv_w] = to_dev(self.dev, torch.zeros(1, 1, ckv_w, D))
                    ttnn.copy(self.ckv if ckv_w == self.nmax else self.ckv[:, :, :ckv_w, :], buf)
                shared.ckv_active = buf
            keys = ttnn.concat([self.win, shared.ckv_active, kv_w], dim=2)
            self.ring_write_multi(kv_w, ctxm)
            return self._attend(q, keys, ctxm.csa_short_mask(r, ckv_w, s), cos, sin)
        keys_w, mask_w = self.window_keys_multi(kv_w, pos, ctxm)
        self.ring_write_multi(kv_w, ctxm)
        if self.mode in ("full", "reindex"):
            reach = ctxm.reach(r)
            masked = ttnn.add(self.index_scores(x, qr, shared.ik, cos, sin), reach)
            kth = ttnn.topk(masked, k=TOPK, dim=-1, largest=True, sorted=True)[0][:, :, :, TOPK - 1 : TOPK]
            shared.mask_multi = ttnn.add(ttnn.mul(ttnn.sub(ttnn.ge(masked, kth), 1.0), -NEG), reach)
        keys = ttnn.concat([keys_w, shared.ckv], dim=2)
        return self._attend(q, keys, ttnn.concat([mask_w, shared.mask_multi], dim=3), cos, sin)

    def restore_pair(self, P):
        """After a rollback to next position P: the pair state is the row of P-1 if P is odd, else nothing."""
        if self.ratio != 2 or self.last_rows is None: return
        kv, sc, g0 = self.last_rows; bk, bs = self.st_bufs()
        if P % 2:
            j = P - 1 - g0
            ttnn.copy(kv[:, :, j : j + 1, :], bk); ttnn.copy(sc[:, :, j : j + 1, :], bs); self.st = (bk, bs)
        else:
            self.st = None

    def decode(self, x, pos, shared: Shared, ctx=None):
        r = self.ratio; n_c = (pos + 1) // r
        self.adopt_state()                                        # no-op unless a prefill just left pair state behind
        if ctx is None:
            ctx = StepCtx(self.dev, pos, self.angles, self.angles, self.nmax if r == 2 else 512, self.nmax if r == 1 else 512)
        cos, sin = ctx.cs_csa
        qr, q = self._q(x, cos, sin)
        from rope_cache_op import cache_write_1
        cache_write_1(self.win, self._kv(x, cos, sin, apply_rope=False), ctx.pos_buf, 0, cos=cos, sin=sin)
        if self.mode == "full":
            kv = mm(x, self.w["comp_wkv"]); latent = None
            if r > 1:
                sc = mm(x, self.w["comp_wgate"]); bk, bs = self.st_bufs()
                if (pos + 1) % r == 0:                                # this token completes a group
                    latent = self._pool(ttnn.concat([bk, kv], dim=2), ttnn.concat([bs, sc], dim=2), 1)
                else:                                                 # first token of a pair: park it at a fixed address
                    ttnn.copy(kv, bk); ttnn.copy(sc, bs); self.st = (bk, bs)
            else:
                latent = kv
            if latent is not None:
                latent = rms_norm(latent, self.w["comp_norm"])
                cos_c, sin_c = ctx.cs_grp2 if r == 2 else ctx.cs_csa                                 # a group sits at its FIRST token's position
                slot_idx = 1 if r == 2 else 2
                ik_raw = rms_norm(mm(latent, self.w["idx_wk"]), self.w["idx_k_norm"])
                cache_write_1(self.ik, ik_raw, ctx.pos_buf, slot_idx, cos=cos_c, sin=sin_c)
                cache_write_1(self.ckv, latent, ctx.pos_buf, slot_idx, cos=cos_c, sin=sin_c)
                if r > 1: self.st = None
            shared.ckv, shared.ik = self.ckv, self.ik
        if n_c <= TOPK and (not shared.uniform or getattr(shared, "fast_short", False)):
            ckv_w = min(self.nmax, max(64, -(-max(n_c, 1) // 64) * 64))
            if self.mode == "full":
                bufs = self.__dict__.setdefault("_ckv_bufs", {})
                buf = bufs.get(ckv_w)
                if buf is None:
                    buf = bufs[ckv_w] = to_dev(self.dev, torch.zeros(1, 1, ckv_w, D))
                    ttnn.copy(self.ckv if ckv_w == self.nmax else self.ckv[:, :, :ckv_w, :], buf)
                elif latent is not None:
                    ttnn.copy(self.ckv if ckv_w == self.nmax else self.ckv[:, :, :ckv_w, :], buf)
                shared.ckv_active = buf
            keys = ttnn.concat([self.win, shared.ckv_active], dim=2)
            return self._attend(q, keys, None, cos, sin, pos_buf=ctx.pos_buf, mode_slot=4 if r == 2 else 5)
        if self.mode in ("full", "reindex"):
            reach = ctx.reach[r]
            if n_c <= TOPK and not shared.uniform:
                mask = reach
            else:
                # n_c <= TOPK inside a trace (shared.uniform): fewer than k reachable entries -> the k-th masked score
                # is NEG -> ge() keeps everything -> mask == reach. Same function, no data-dependent branch.
                masked = ttnn.add(self.index_scores(x, qr, shared.ik, cos, sin), reach)
                kth = ttnn.topk(masked, k=TOPK, dim=-1, largest=True, sorted=True)[0][:, :, :, TOPK - 1 : TOPK]
                mask = ttnn.add(ttnn.mul(ttnn.sub(ttnn.ge(masked, kth), 1.0), -NEG), reach)
            buf = shared.mask_bufs.get(r)
            if buf is None:
                buf = shared.mask_bufs[r] = to_dev(self.dev, torch.zeros(1, 1, 1, self.nmax))
            ttnn.copy(mask, buf); shared.mask_dec = buf
        keys = ttnn.concat([self.win, shared.ckv], dim=2)
        return self._attend(q, keys, ttnn.concat([ctx.ring_mask, shared.mask_dec], dim=3), cos, sin)
