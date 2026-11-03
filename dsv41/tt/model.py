"""The whole model on the mesh: traced decode, experts paged from the packed store.

HOST does bytes, bookkeeping, and a few small steps: embedding rows, Engram hash ids + FP8 rows, the
router's softplus, sqrt, top-6, and weight normalization, expert paging, token selection. DEVICE does
everything else in the forward pass.
Prefill here runs ALL 40 layers over every prompt token, like the reference (exact).
"""
import gc
import os
import time
import numpy as np
import torch
import ttnn
from ckpt import Checkpoint
from mesh import to_dev
from weights import Residents
from layer import Layer
from mhc import MHC
from csa import Shared
from attn import StepCtx, MultiCtx
from moe import Slots, ExpertPool
from store import ExpertStore
from engram import EngramHost, Engram
from head import Head
from sinkhorn_op import SinkhornOp
from slot_cat_op import SlotCatOp

N_LAYERS, HC, DIM = 40, 4, 5120


class Model:
    def __init__(self, dev, max_ctx=2048, n_layers=N_LAYERS, store_fmt="mixed", log=print, pool_slots=None):
        self.dev, self.n_layers, self.log = dev, n_layers, log
        self.ck = Checkpoint(); self.R = Residents(dev, self.ck); self.store = ExpertStore(store_fmt)
        fmts = {self.store.fmt(L) for L in range(n_layers)}
        self.slots = {f: Slots(dev, ttnn.bfloat4_b if f == "bfp4" else ttnn.bfloat8_b) for f in fmts}
        self.sink_op = None if os.environ.get("DSV41_NO_SINKOP") == "1" else SinkhornOp(dev)   # dev knob: stock-op Sinkhorn
        self.slot_cat = SlotCatOp(dev)
        # resident expert pools (decode only), one per format: {"bfp8": n, "bfp4": n}
        self.pools = {f: ExpertPool(dev, ttnn.bfloat4_b if f == "bfp4" else ttnn.bfloat8_b, n) for f, n in (pool_slots or {}).items() if n}
        self.layers = [Layer(dev, self.R, L, max_pos=max_ctx, slots=self.slots[self.store.fmt(L)], sink_op=self.sink_op, slot_cat=self.slot_cat) for L in range(n_layers)]
        self._ring_snaps = [to_dev(dev, torch.zeros(1, 1, 128, 512)) for _ in range(n_layers)]
        self.engram_host = EngramHost(self.ck, os.path.join(os.environ["DSV41_ROOT"], "weights", "engram_hash.npz"))
        self.engram = {L: Engram(dev, self.layers[L].w) for L in (1, 14) if L < n_layers}
        self.head = Head(dev, self.R.top(), sink_op=self.sink_op)
        eb, _, es = self.ck.raw("embed.weight")
        self.embed = torch.from_numpy(eb).view(torch.bfloat16).view(es)                 # mmap: row lookup on the host
        self.shared = Shared()
        self.where = "init"                 # progress marker, for naming the step when a run hangs
        self.traced, self._trace = False, None   # s18: decode through captured device traces (needs a trace region)
        self.spark_targets, self.drafter = (), None   # s19: set by enable_dspark()
        log(f"[model] {n_layers} layers on the mesh, residents {self.R.bytes/1e9:.2f} GB/chip, slot formats {sorted(fmts)}")

    def _embed(self, ids):
        rows = self.embed[torch.as_tensor(ids, dtype=torch.long)].float()              # [s, d]
        one = to_dev(self.dev, rows.view(1, 1, rows.shape[0], DIM))                     # upload once, replicate on device
        return ttnn.concat([one] * HC, dim=1)                                           # [1,4,s,d]

    def _pre0(self, s):
        m = torch.zeros(1, HC, s, 1); m[:, 0] = 1.0                                     # identity pre-mix: copy 0
        return to_dev(self.dev, m)

    def enable_dspark(self):
        from dspark import Drafter, TARGETS
        self.spark_targets = TARGETS
        if "bfp4" not in self.slots:
            self.slots["bfp4"] = Slots(self.dev, ttnn.bfloat4_b)
        self.drafter = Drafter(
            self.dev,
            self.R,
            self.store,
            self.embed,
            self.head,
            max_pos=self.layers[0].attn.angles.shape[0],
            sink_op=self.sink_op,
            slots=self.slots["bfp4"],
            slot_cat=self.slot_cat,
        )
        self.log(f"[dspark] drafter up: 3 blocks, 384 resident experts, targets {TARGETS}")

    def _mh_take(self, L, x, s, mh_parts):
        """Prefill hook: keep the mean over hc of the target layers' inputs for the last 128 positions."""
        if L in self.spark_targets:
            m = ttnn.mean(x, dim=1, keepdim=True)
            mh_parts.append(m if s <= 128 else m[:, :, s - 128 :, :])

    def _seed_drafter(self, mh_parts, start_pos):
        if self.drafter is not None and mh_parts:
            self.drafter.seed(ttnn.concat(mh_parts, dim=3), start_pos)

    def verify(self, tokens):
        """Run s tokens at pos.. through the decode semantics in one pass (in-place state). Returns logits [s,V],
        main_hidden [1,1,s,3d], and a rollback handle (ring snapshots); pos/ids advance by s."""
        s, pos = len(tokens), self.pos
        x, pre = self._embed(tokens), self._pre0(s)
        hid = self.engram_host.hash_ids(tokens, pos)
        ctxm = MultiCtx(self.dev, pos, s, self.layers[0].attn.angles, self.layers[2].attn.angles, self.layers[2].attn.nmax, self.layers[20].attn.nmax)
        if pos + s > 128:
            for l, snap in zip(self.layers, self._ring_snaps):
                ttnn.copy(l.attn.win, snap)
        rings = self._ring_snaps
        mh_parts = []; prof = getattr(self, "vprof", None)
        for L, layer in enumerate(self.layers):
            if L in self.engram:
                x = self.engram[L].forward(x, self.engram_host.rows(L, hid[:, (1, 14).index(L)]))
            self.where = f"verify pos={pos} s={s} L{L}"
            x, pre, mh = layer.decode_multi(x, pre, pos, self.shared, self.store.get, ctxm, self.pools.get(self.store.fmt(L)), want_mh=L in self.spark_targets, prof=prof)
            if mh is not None: mh_parts.append(mh)
        if prof is not None: ttnn.synchronize_device(self.dev); t0 = time.perf_counter()
        out = self.head.logits(x, pre, last_only=False)
        if prof is not None: prof["head"] += time.perf_counter() - t0
        self.pos = pos + s; self.ids += list(tokens)
        return out, ttnn.concat(mh_parts, dim=3), (pos, s, rings)

    def rollback(self, handle, accepted):
        """Keep the first `accepted` of the verified tokens: restore the rejected ring slots from the snapshot,
        the ratio-2 pair state, and the host-side histories."""
        pos, s, rings = handle; P = pos + accepted
        if accepted < s:
            if pos + s > 128:
                rej = torch.zeros(1, 1, 128, 1)
                for i in range(accepted, s): rej[0, 0, (pos + i) % 128, 0] = 1.0
                rej, keep = to_dev(self.dev, rej), to_dev(self.dev, 1.0 - rej)
                for l, saved in zip(self.layers, rings):
                    ttnn.copy(ttnn.add(ttnn.mul(l.attn.win, keep), ttnn.mul(saved, rej)), l.attn.win)
            drop = s - accepted
            self.ids = self.ids[:-drop]; self.engram_host.hist = self.engram_host.hist[:-drop]
        for l in self.layers:
            if l.mode == "full" and l.ratio == 2: l.attn.restore_pair(P)
        self.pos = P

    def spec_warm(self, k):
        """Run the verify (s = k+1) and draft graphs once at the current parity and undo them: every program is
        compiled and every lazily allocated op state exists BEFORE the decode traces are captured (the trace
        hazard, see DecodeTrace). Rings, pair state, positions and histories are restored exactly."""
        snap = self.drafter.snapshot()
        self.drafter.draft(0, None, max(self.pos - 1, 0), k=k, want_conf=False)
        for R in (1, 32):                                                   # both main-hidden row counts the loop uses
            z_mh = to_dev(self.dev, torch.zeros(1, 1, R, 3 * DIM))
            self.drafter.append(z_mh, self.pos, 1)
            self.drafter.draft(0, z_mh, max(self.pos - 1, 0), k=k); self.drafter.restore(snap)
        _, mh_all, handle = self.verify([0] * (k + 1)); self.rollback(handle, 0)
        self.drafter.append(mh_all, self.pos, 1); self.drafter.draft(0, self.drafter.rows(mh_all, 0, 1), max(self.pos - 1, 0), k=k); self.drafter.restore(snap)

    def generate_spec(self, first_logits, n, k=None, emit=None):
        """Greedy generation with DSpark drafts verified in one multi-token pass each. `first_logits` = logits of
        the prompt's last position (from prefill); yields exactly the tokens plain greedy decoding would produce
        up to bf16 path noise. Returns (tokens, stats). Requires traced decode (the main hidden comes from the
        trace buffers)."""
        from dspark import BLOCK
        assert self.traced and self.drafter is not None
        k = BLOCK if k is None else k
        out, stats = [], {"passes": 0, "verified": 0, "accepted_drafts": 0, "plain": 0}
        def plain(tok):
            lg = self.decode(tok); stats["plain"] += 1
            # the reference drafts after EVERY main step, which also files that position's main key in the drafter's
            # ring; a plain step here must file it too or the ring has holes and the drafts degrade
            self.drafter.append(ttnn.concat([self._trace.bufs[L]["mh"] for L in self.spark_targets], dim=3), self.pos - 1)
            return lg
        nxt = int(first_logits[-1].argmax()); out.append(nxt)
        if emit: emit(nxt)
        if self._trace is None or not self._trace.tid:                     # first generation after a prefill: warm both parities, capture
            self.spec_warm(k); lg = plain(nxt); self.spec_warm(k)
            for _ in range(2):
                nxt = int(lg[-1].argmax()); out.append(nxt)
                if emit: emit(nxt)
                lg = plain(nxt)
        else:
            lg = plain(nxt)                                                  # the reference drafts only after a decode step
        cur = int(lg[-1].argmax())
        T = stats.setdefault("t", {"draft": 0.0, "verify": 0.0, "rollback": 0.0, "append": 0.0})
        while len(out) < n:
            t0 = time.perf_counter()
            drafts, _, _ = self.drafter.draft(cur, None, self.pos - 1, k=k, want_conf=False)
            toks = [cur] + drafts[:k]; t1 = time.perf_counter(); T["draft"] += t1 - t0
            logits, mh_all, handle = self.verify(toks); stats["passes"] += 1; stats["verified"] += len(toks)
            a = 1
            for i in range(len(toks) - 1):
                if int(logits[i].argmax()) == toks[i + 1]: a += 1
                else: break
            bonus = int(logits[a - 1].argmax()); t2 = time.perf_counter(); T["verify"] += t2 - t1
            if os.environ.get("DSV41_SPEC_TRACE"): self.log(f"[spec] pos {handle[0]} toks {toks} argmaxes {[int(logits[i].argmax()) for i in range(len(toks))]} accepted {a} bonus {bonus}")
            self.rollback(handle, a); t3 = time.perf_counter(); T["rollback"] += t3 - t2
            self.drafter.append(mh_all, handle[0], a); T["append"] += time.perf_counter() - t3
            for t in toks[:a]:
                out.append(t)
                if emit: emit(t)
                if len(out) >= n: break
            stats["accepted_drafts"] += a - 1
            cur = bonus
        return out[:n], stats

    def _rows_with_images(self, ids, types, images):
        """Embedding rows with each image span overwritten: delimiters by their learned embeddings, IMAGE slots by the
        ViT+aligner features (computed on the mesh, merged here: bytes). types: -1 text, 0 start, 1 image, 2 newline, 3 end."""
        from vision import Vision
        if not hasattr(self, "vision"): self.vision = Vision(self.dev, self.R)
        rows = self.embed[torch.as_tensor(ids, dtype=torch.long)].float(); types = torch.as_tensor(types)
        special = {0: self.R.torch("image_start"), 2: self.R.torch("image_newline"), 3: self.R.torch("image_end")}
        for im in images:
            sl = slice(im["start"], im["start"] + len(im["types"])); t = torch.as_tensor(im["types"]); span = rows[sl]
            for k, v in special.items(): span[t == k] = v
            self.vit_out, emb = self.vision.encode(im["patches"], im["n_vit_h"], im["n_vit_w"]); span[t == 1] = emb; self.image_embeds = emb
            rows[sl] = span
        return rows, types >= 0

    def prefill(self, ids, all_logits=False, pass_size=8192, images=None, token_types=None):
        """Prefill in passes of `pass_size` tokens (a multiple of 128). Each pass streams every used expert once
        (~13 s fixed), so passes should be as large as memory allows; later passes continue from the rings and
        the compressed caches left by earlier ones."""
        assert pass_size % 128 == 0
        out = []
        for start in range(0, len(ids), pass_size):
            chunk_ids = ids[start : start + pass_size]; s = len(chunk_ids)
            imask, mh_parts, last = None, [], start + len(chunk_ids) >= len(ids)
            if images:                                        # image spans must sit inside the first pass (as in the reference)
                assert start == 0 and len(ids) <= pass_size
                rows, imask = self._rows_with_images(chunk_ids, token_types, images)
                one = to_dev(self.dev, rows.view(1, 1, s, DIM)); x, pre = ttnn.concat([one] * HC, dim=1), self._pre0(s)
            else:
                x, pre = self._embed(chunk_ids), self._pre0(s)
            self.shared = Shared()                            # masks and position caches are per pass; KV lives in the layers
            hid = self.engram_host.hash_ids(chunk_ids, start, None if imask is None else (~imask).numpy())
            for L, layer in enumerate(self.layers):
                if L in self.engram:
                    x = self.engram[L].forward(x, self.engram_host.rows(L, hid[:, (1, 14).index(L)]), None if imask is None else ~imask)
                self.where = f"prefill pass@{start} L{L} s={s}"
                self._mh_take(L, x, s, mh_parts) if last else None
                x_in = x
                x, pre = layer.prefill(x, pre, self.shared, self.store.get, self.pools.get(self.store.fmt(L)), start_pos=start, image_mask=imask)
                if s > 1024: ttnn.deallocate(x_in); gc.collect()
            if all_logits or last:
                out.append(self.head.logits(x, pre, last_only=not all_logits))
            if last: self._seed_drafter(mh_parts, start + max(0, s - 128))
        self.pos, self.ids = len(ids), list(ids)
        return torch.cat(out, 0) if len(out) > 1 else out[0]

    def decode(self, token_id, want_logits=True):
        if self.traced:
            if self._trace is None or self._trace.stale():
                if self._trace is not None: self._trace.release()      # a prefill replaced the state tensors
                self._trace = DecodeTrace(self)
            return self._trace.step(token_id, want_logits=want_logits)
        pos = self.pos; x, pre = self._embed([token_id]), self._pre0(1)
        hid = self.engram_host.hash_ids([token_id], pos)
        ctx = StepCtx(self.dev, pos, self.layers[0].attn.angles, self.layers[2].attn.angles, self.layers[2].attn.nmax, self.layers[20].attn.nmax)
        for L, layer in enumerate(self.layers):
            if L in self.engram:
                x = self.engram[L].forward(x, self.engram_host.rows(L, hid[:, (1, 14).index(L)]))
            self.where = f"decode pos={pos} L{L}"
            x, pre = layer.decode(x, pre, pos, self.shared, self.store.get, ctx, self.pools.get(self.store.fmt(L)))
        self.pos = pos + 1; self.ids.append(token_id)
        return self.head.logits(x, pre) if want_logits else None


    # ---- persistent prefix state (s16) --------------------------------------------------------------------
    # What a prefix IS on this engine: 40 window rings [128,512] (5 MB total - so we simply keep them; the
    # report's SWA Bounded Replay exists because THEIR per-session SWA KV is large, ours is not), the four
    # KV-source layers' compressed-KV and index-key caches up to n_c, the ratio-2 compressors' pending pair
    # state, and the token ids (Engram hashing needs the history). Restore is exact: no replay, no approximation.

class DecodeTrace:
    """s18: the device half of every layer's decode step (Layer.decode_a: mHC, attention, mHC, router scores) is
    captured ONCE as a device trace and replayed every token; the expert half (Layer.decode_b) stays eager on
    whichever pool slots hold the six experts. What makes a trace legal here: every per-token input lands at a
    fixed address (StepCtx.update, the Engram rows buffer, the X/PRE stream buffers), every piece of layer state
    is updated in place (rings, compressed caches, pair state, the published selection mask), no op depends on
    the position (Shared.uniform), and every trace writes its results into buffers allocated BEFORE any capture.

    TRAP (cost a morning): tt-metal warns "allocating device buffers is unsafe due to the existence of an active
    trace" and means it. Anything allocated after a capture (a later trace's kept outputs, an op's lazily created
    persistent state such as a reshape mapping tensor on a graph's first eager run) can land where that trace's
    freed intermediates live; the next replay overwrites it and the next EAGER op that reads it hangs the device.
    So: two eager tokens first (every graph variant runs, both pair parities, every persistent thing exists),
    then ALL traces are captured in one go, outputs copied into pre-allocated buffers, and from then on nothing
    new is allocated for the life of the traces. Any prefill replaces state tensors -> stale()."""
    WARM = 2

    def __init__(self, m: "Model"):
        self.m, dev = m, m.dev
        self.x_in, self.pre_in = to_dev(dev, torch.zeros(1, HC, 1, DIM)), to_dev(dev, torch.zeros(1, HC, 1, 1))
        self.embed_rm_in = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, 1, DIM]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG
        )
        self._rep = ttnn.ReplicateTensorToMesh(dev)
        self.ctx = StepCtx(dev, m.pos, m.layers[0].attn.angles, m.layers[2].attn.angles, m.layers[2].attn.nmax, m.layers[20].attn.nmax)
        for L in m.engram:
            m.engram[L].put_rows(torch.zeros(1, m.engram[L].wkv.shape[2]))
        for l in m.layers:
            if l.mode == "full":
                n_c = (m.pos + 1) // l.attn.ratio
                if n_c <= 512:
                    ckv_w = min(l.attn.nmax, max(64, -(-max(n_c, 1) // 64) * 64))
                    bufs = l.attn.__dict__.setdefault("_ckv_bufs", {})
                    buf = bufs.get(ckv_w)
                    if buf is None:
                        buf = bufs[ckv_w] = to_dev(dev, torch.zeros(1, 1, ckv_w, 512))
                    ttnn.copy(l.attn.ckv if ckv_w == l.attn.nmax else l.attn.ckv[:, :, :ckv_w, :], buf)
        m.shared.uniform = True
        m.shared.fast_short = True
        # one output buffer set per layer, allocated now (before any capture)
        z = lambda *shp, dt=ttnn.bfloat16: to_dev(dev, torch.zeros(*shp), dtype=dt)
        comb_shp = (1, 1, HC, HC) if (m.sink_op is not None and hasattr(m.sink_op, "mixes_1")) else (1, HC, 1, HC)
        self.bufs = [{"h": z(1, 1, 1, DIM), "sc": z(1, 1, 1, 384, dt=ttnn.float32), "x": z(1, HC, 1, DIM),
                      "f_pre": z(1, HC, 1, 1), "f_post": z(1, HC, 1, 1), "f_comb": z(*comb_shp)} for _ in m.layers]
        for L in m.spark_targets:                                   # DSpark reads these layers' inputs (mean over hc)
            self.bufs[L]["mh"] = z(1, 1, 1, DIM)
        self.wts = z(1, 1, 1, 6 * 288)                              # this layer's six routing weights expanded across 288 cols/chip (populated by SlotCatOp)
        self.slot_cat = m.slot_cat
        self.sig = self._sig()
        self.n_warm, self.tid = 0, {}
        self.n_eager = self.n_capture = self.n_replay = 0
        self.trace_layers = set(range(len(m.layers)))            # dev knob: layers outside this set run decode_a eagerly

    def _sig(self):
        m = self.m
        return (
            (id(m.shared), (m.pos + 1) > 512, (m.pos + 1) > 1024, (m.pos // 64) if (m.pos + 1) <= 512 else -1)
            + tuple(id(l.attn.win) for l in m.layers)
            + tuple((id(l.attn.ckv), id(l.attn.ik)) for l in m.layers if l.mode == "full")
        )

    def stale(self):
        return self._sig() != self.sig

    def release(self):
        for tid in self.tid.values():
            ttnn.release_trace(self.m.dev, tid)
        self.tid = {}

    def _key(self, L, pos):
        l = self.m.layers[L]
        return (L, pos % 2 if (l.mode == "full" and l.ratio == 2) else 0)

    def _run(self, L, pos):
        m = self.m
        if L == 0:
            from rope_cache_op import tilize_row0_1
            tilize_row0_1(self.embed_rm_in, self.x_in, pre_dst=self.pre_in)
        x = m.engram[L].forward(self.x_in, m.engram[L].rows1) if L in m.engram else self.x_in
        defer = m.sink_op is not None and hasattr(m.sink_op, "post_pre_ms")
        return m.layers[L].decode_a(x, self.pre_in, pos, m.shared, self.ctx, want_mh=L in m.spark_targets, defer_ffn_mix=defer)

    def _capture_all(self, pos):
        """Every layer, both pair parities where they matter, in layer order (consumers read what sources publish
        through `shared` at capture time, exactly as in eager order). Nothing executes here."""
        m, dev = self.m, self.m.dev
        for L in range(len(m.layers)):
            if L not in self.trace_layers: continue
            for p in ((pos, pos + 1) if self._key(L, pos) != self._key(L, pos + 1) else (pos,)):
                key = self._key(L, p)
                tid = ttnn.begin_trace_capture(dev, cq_id=0)
                a = self._run(L, p)
                for k, t in a.items():
                    ttnn.copy(t, self.bufs[L][k])
                ttnn.end_trace_capture(dev, tid, cq_id=0)
                del a                                              # trace intermediates: free, never referenced again
                self.tid[key] = tid; self.n_capture += 1
            # the expert half: six FIXED slot tensors (the layer format's Slots) that the eager step fills by D2D copy
            self.slot_cat.update_ctrl(m.slots[m.store.fmt(L)].s, torch.zeros(6))
            tid = ttnn.begin_trace_capture(dev, cq_id=0)
            self._run_b(L)
            ttnn.end_trace_capture(dev, tid, cq_id=0)
            self.tid[("B", L)] = tid; self.n_capture += 1

    def _run_b(self, L):
        """Experts on the concatenated 6-expert buffers + mHC post, writing the next layer's stream buffers in place."""
        m, layer, b = self.m, self.m.layers[L], self.bufs[L]
        fmt = m.store.fmt(L)
        if m.sink_op is not None and hasattr(m.sink_op, "post_only"):
            self.slot_cat.run_from_ctrl(fmt, m.slots[fmt].cat)
            h = layer.moe.decode(b["h"], None, None, cat=m.slots[fmt].cat, slot_cat=self.slot_cat)
            f_pre, f_post, f_comb = layer.hc_ffn.mixes(m.sink_op.x_mid, ms=m.sink_op.ms_buf)
            m.sink_op.post_only(h, m.sink_op.x_mid, f_post, f_comb, x_out=self.x_in)
            ttnn.copy(f_pre, self.pre_in)
        else:
            self.slot_cat.run_from_ctrl(fmt, m.slots[fmt].cat, self.wts)
            h = layer.moe.decode(b["h"], None, None, cat=m.slots[fmt].cat, wts_dev=self.wts)
            ttnn.copy(MHC.post(h, b["x"], b["f_post"], b["f_comb"]), self.x_in)
            ttnn.copy(b["f_pre"], self.pre_in)

    def _b(self, L):
        m, dev, key = self.m, self.m.dev, ("B", L)
        if key in self.tid:
            ttnn.execute_trace(dev, self.tid[key], cq_id=0, blocking=False); self.n_replay += 1
        else:
            self.n_eager += 1; self._run_b(L)

    def _a(self, L, pos):
        m, dev, key = self.m, self.m.dev, self._key(L, pos)
        if key in self.tid:
            ttnn.execute_trace(dev, self.tid[key], cq_id=0, blocking=False); self.n_replay += 1
            return self.bufs[L]
        self.n_eager += 1
        a = self._run(L, pos)
        for k, t in a.items():                                     # same buffers either way: decode_b never sees a difference
            ttnn.copy(t, self.bufs[L][k])
        return self.bufs[L]

    def step(self, token_id, want_logits=True):
        m, dev, pos = self.m, self.m.dev, self.m.pos
        if not self.tid and self.n_warm >= self.WARM:
            self._capture_all(pos)
        self.ctx.update(pos)
        row = m.embed[token_id].view(1, 1, 1, DIM).contiguous()
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(row, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self._rep),
            self.embed_rm_in,
        )
        hid = m.engram_host.hash_ids([token_id], pos)
        for L in m.engram:
            m.engram[L].put_rows(m.engram_host.rows(L, hid[:, (1, 14).index(L)]))
        for l in m.layers:
            if l.mode == "full" and l.ratio == 2:
                l.attn.adopt_state()
        prof = getattr(self, "prof", None)                       # dev: per-component device time (syncs added)
        for L, layer in enumerate(m.layers):
            m.where = f"decode(traced) pos={pos} L{L}"
            if prof is not None: ttnn.synchronize_device(dev); t0 = time.perf_counter()
            a = self._a(L, pos)
            if prof is not None: ttnn.synchronize_device(dev); t1 = time.perf_counter(); prof["a"] += t1 - t0
            idx, wt, _ = layer.moe.pick(a["sc"])
            if prof is not None: t2 = time.perf_counter(); prof["pick"] += t2 - t1
            fmt = m.store.fmt(L); pool, slot_obj = m.pools.get(fmt), m.slots[fmt]
            keys = [(L, int(e)) for e in idx[0]]
            if pool is not None:
                slots = pool.ensure(keys, m.store.get)
                if prof is not None: ttnn.synchronize_device(dev); t3 = time.perf_counter(); prof["ensure"] += t3 - t2
            else:
                for k in range(6):
                    layer.moe.stream(k, m.store.get(*keys[k]))
                slots = slot_obj.s
                if prof is not None: ttnn.synchronize_device(dev); t3 = time.perf_counter(); prof["ensure"] += t3 - t2
            self.slot_cat.update_ctrl(slots, wt[0])
            if prof is not None: ttnn.synchronize_device(dev); t4 = time.perf_counter(); prof["slot copies+wts"] += t4 - t3
            self._b(L)
            if prof is not None: ttnn.synchronize_device(dev); prof["experts+post (B)"] += time.perf_counter() - t4
        for l in m.layers:                         # Python-side view of the pair state a replay leaves behind
            if l.mode == "full" and l.ratio == 2:
                l.attn.st = l.attn.st_bufs() if pos % 2 == 0 else None
        m.pos = pos + 1; m.ids.append(token_id); self.n_warm += 1
        return m.head.logits(self.x_in, self.pre_in) if want_logits else None

    def dram_gb(self):
        try: return ttnn.get_memory_view(self.m.dev, ttnn.BufferType.DRAM).total_bytes_allocated_per_bank * 12 / 1e9
        except Exception: return float("nan")
