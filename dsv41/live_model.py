"""40-layer DeepSeek-V4.1-Flash decode engine on the 1x8 Wormhole B0 mesh."""
import glob
import os
import sys
import time


def _ensure_paths():
    dsv_root = os.environ.get("DSV41_ROOT") or os.path.dirname(os.path.abspath(__file__))
    os.environ["DSV41_ROOT"] = dsv_root
    for sub in ("tt", "", "ref"):
        p = os.path.join(dsv_root, sub) if sub else dsv_root
        if p not in sys.path:
            sys.path.insert(0, p)
    return dsv_root


def get_tokenizer():
    from transformers import AutoTokenizer
    snap_glob = os.environ.get("DSV41_SNAP", "~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/*")
    matches = sorted(glob.glob(os.path.expanduser(snap_glob)))
    return AutoTokenizer.from_pretrained(matches[-1])


class Live40LEngine:
    """Wraps the 40-layer DeepSeek-V4.1-Flash Model on the 1x8 Wormhole mesh (with optional DSpark speculative fast path)."""

    def __init__(self, max_ctx=1024, trace_region_mb=64, pool_bfp8=None, pool_bfp4=None, traced=True, use_dspark=None):
        _ensure_paths()
        from mesh import open_mesh, close_mesh
        from model import Model
        self._close_mesh = close_mesh
        if use_dspark is None:
            use_dspark = os.environ.get("MK_DSPARK", "0") == "1"
        self.use_dspark = bool(use_dspark)
        if pool_bfp8 is None:
            pool_bfp8 = int(os.environ.get("MK_POOL8", 935 if self.use_dspark else 950))
        if pool_bfp4 is None:
            pool_bfp4 = int(os.environ.get("MK_POOL4", 1020 if self.use_dspark else 1500))
        t0 = time.perf_counter()
        self.dev = open_mesh(trace_region_mb=trace_region_mb)
        self.model = Model(
            self.dev,
            max_ctx=max_ctx,
            pool_slots={"bfp8": pool_bfp8, "bfp4": pool_bfp4},
        )
        self.model.traced = traced
        if self.use_dspark:
            self.model.enable_dspark()
        self.tok = get_tokenizer()
        self._last_tok = 0
        self._last_logits = None
        if traced:
            # Seed persistent state buffers (ckv, ik, win, shared) and capture DecodeTrace at startup
            _ = self.model.prefill([0, 0], pass_size=128)
            k_warm = int(os.environ.get("MK_DSPARK_K", 1))
            warm_ks = tuple(int(x) for x in os.environ.get("MK_DSPARK_WARM_KS", str(k_warm)).split(",") if x.strip())
            if self.use_dspark:
                for kw in warm_ks:
                    self.model.spec_warm(kw)
            _ = self.model.decode(0, want_logits=False)
            if self.use_dspark:
                for kw in warm_ks:
                    self.model.spec_warm(kw)
            for _ in range(2):
                _ = self.model.decode(0, want_logits=False)
            self._reset_state_keep_trace(0)
            for t in (int(self.tok.bos_token_id or 0), 128803):
                _ = self.model.decode(t, want_logits=False)
                self._append_drafter_last()
        self.load_s = time.perf_counter() - t0

    def _append_drafter_last(self):
        if getattr(self.model, "drafter", None) is not None and getattr(self.model, "_trace", None) is not None:
            import ttnn
            self.model.drafter.append(
                ttnn.concat([self.model._trace.bufs[L]["mh"] for L in self.model.spark_targets], dim=3),
                self.model.pos - 1,
            )

    def enable_dspark(self):
        if getattr(self.model, "drafter", None) is None:
            if getattr(self.model, "_trace", None) is not None:
                try:
                    self.model._trace.release()
                except Exception:
                    pass
                self.model._trace = None
            self.model.enable_dspark()
            self.use_dspark = True

    def _reset_state_keep_trace(self, pos: int = 0):
        self.model.pos = int(pos)
        self.model.ids = self.model.ids[:pos] if pos > 0 else []
        self.model.engram_host.hist = self.model.engram_host.hist[:pos] if pos > 0 else []
        for l in self.model.layers:
            if l.mode == "full" and l.ratio == 2:
                l.attn.st = None

    def prefill_or_extend(self, ids: list[int], reset: bool = True) -> tuple[int, float, float]:
        t0 = time.perf_counter()
        prompt_ids = [int(x) for x in ids]
        bos_id = int(self.tok.bos_token_id or 0)
        if not reset and getattr(self.model, "pos", 0) > 0:
            prev_pos = int(self.model.pos)
            lg = None
            for t in prompt_ids:
                lg = self.model.decode(int(t))
                self._append_drafter_last()
            hit = prev_pos / max(1, prev_pos + len(prompt_ids))
        elif (
            getattr(self.model, "_trace", None) is not None
            and not self.model._trace.stale()
            and len(prompt_ids) <= 126
        ):
            if not prompt_ids or prompt_ids[0] != bos_id:
                prompt_ids = [bos_id] + prompt_ids
            common = 0
            if len(self.model.ids) <= 128:
                max_c = min(len(self.model.ids), max(0, len(prompt_ids) - 1))
                while common < max_c and self.model.ids[common] == prompt_ids[common]:
                    common += 1
                common = (common // 2) * 2
            self._reset_state_keep_trace(common)
            lg = None
            rem = prompt_ids[common:]
            for idx, t in enumerate(rem):
                lg = self.model.decode(int(t), want_logits=(idx == len(rem) - 1))
                self._append_drafter_last()
            hit = common / max(1, len(prompt_ids))
        else:
            if len(prompt_ids) < 2 or prompt_ids[0] != bos_id:
                prompt_ids = [bos_id] + prompt_ids
            if getattr(self.model, "_trace", None) is not None:
                try:
                    self.model._trace.release()
                except Exception:
                    pass
                self.model._trace = None
            self.model.pos = 0
            self.model.ids = []
            self.model.engram_host.hist = []
            self.model.shared = type(self.model.shared)()
            lg = self.model.prefill(prompt_ids)
            hit = 0.0
        nxt = int(lg[-1].argmax())
        self._last_tok = nxt
        self._last_logits = lg
        return nxt, time.perf_counter() - t0, float(hit)

    def step_decode(self) -> int:
        lg = self.model.decode(int(self._last_tok))
        nxt = int(lg[-1].argmax())
        self._last_tok = nxt
        self._last_logits = lg
        return nxt

    def generate_spec(self, prompt_ids: list[int], max_new_tokens: int = 16, k: int = 1, emit=None):
        """DSpark speculative multi-token generation with target verification and rollback."""
        self.enable_dspark()
        _, prefill_s, _ = self.prefill_or_extend(list(prompt_ids), reset=True)
        t_dec = time.perf_counter()
        out_ids, stats = self.model.generate_spec(self._last_logits, max_new_tokens, k=k, emit=emit)
        decode_s = time.perf_counter() - t_dec
        if out_ids:
            self._last_tok = int(out_ids[-1])
        text = self.tok.decode(out_ids)
        return {
            "prompt_ids": list(prompt_ids),
            "out_ids": [int(x) for x in out_ids],
            "text": text,
            "prefill_s": round(prefill_s, 3),
            "decode_s": round(decode_s, 3),
            "spec_stats": stats,
            "load_s": round(self.load_s, 3),
        }

    def generate_text(self, prompt_text: str, max_new_tokens: int = 16):
        ids = [self.tok.bos_token_id] + self.tok.encode(prompt_text, add_special_tokens=False)
        return self.generate_ids(ids, max_new_tokens=max_new_tokens)

    def generate_ids(self, prompt_ids: list[int], max_new_tokens: int = 16):
        nxt, prefill_s, _ = self.prefill_or_extend(list(prompt_ids), reset=True)
        out_ids = []
        step_times = []
        if max_new_tokens > 0:
            out_ids.append(int(nxt))
            step_times.append(0.0)
        for _ in range(1, max_new_tokens):
            ts = time.perf_counter()
            nxt = self.step_decode()
            out_ids.append(int(nxt))
            step_times.append(time.perf_counter() - ts)
        text = self.tok.decode(out_ids)
        return {
            "prompt_ids": list(prompt_ids),
            "out_ids": out_ids,
            "text": text,
            "prefill_s": round(prefill_s, 3),
            "step_times_s": [round(x, 3) for x in step_times],
            "load_s": round(self.load_s, 3),
        }

    def close(self):
        if self.dev is not None:
            self._close_mesh(self.dev)
            self.dev = None
