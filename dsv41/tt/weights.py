"""Streamed checkpoint loader -> device tensors on the 1x8 mesh.

Supports transparent pre-tilized .tensorbin + host .pt caching under
$DSV41_ROOT/weights/residents_cache
so subsequent 40-layer model boots skip FP8 CPU dequantization, transpose, and
CPU tilizing, loading directly into 1x8 Wormhole DRAM via ttnn.load_tensor.
"""
import os
import numpy as np
import torch
import ttnn
from ckpt import Checkpoint
from mesh import to_dev, rep, shard


def _e4m3_lut():
    lut = np.empty(256, dtype=np.float32)
    for b in range(256):
        s, e, m = b >> 7, (b >> 3) & 0xF, b & 0x7
        v = (m / 8.0) * 2.0 ** -6 if e == 0 else (1 + m / 8.0) * 2.0 ** (e - 7)
        lut[b] = 0.0 if (e == 0xF and m == 0x7) else (-v if s else v)
    return lut


E4M3 = _e4m3_lut()


def _resolve_cache_dir() -> str:
    root = os.environ.get("DSV41_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cdir = os.path.join(root, "weights", "residents_cache")
    os.makedirs(cdir, exist_ok=True)
    return cdir


class Residents:
    def __init__(self, dev, ck: Checkpoint = None):
        self.dev, self.ck = dev, ck or Checkpoint()
        self.bytes = 0
        self.cache_dir = _resolve_cache_dir()
        self.use_cache = os.environ.get("DSV41_NO_RES_CACHE", "0") != "1"

    def has(self, name):
        return name in self.ck.weight_map

    def _raw_torch(self, name) -> torch.Tensor:
        buf, dt, shape = self.ck.raw(name)
        if dt == "F8_E4M3":
            w = E4M3[buf].reshape(shape)
            sname = name[: -len("weight")] + "scale" if name.endswith("weight") else None
            if sname and self.has(sname):
                sb, sd, ss = self.ck.raw(sname)
                assert sd == "F8_E8M0", sd
                scale = np.ldexp(np.float32(1.0), sb.reshape(ss).astype(np.int32) - 127)
                bo, bi = -(-shape[0] // ss[0]), -(-shape[1] // ss[1])
                w = w * np.repeat(np.repeat(scale, bo, 0)[: shape[0]], bi, 1)[:, : shape[1]]
            return torch.from_numpy(np.ascontiguousarray(w))
        if dt == "BF16":
            return torch.from_numpy(buf.copy()).view(torch.bfloat16).view(shape).float()
        if dt == "F32":
            return torch.from_numpy(buf.copy()).view(torch.float32).view(shape)
        raise ValueError(f"{name}: unhandled dtype {dt}")

    def torch(self, name) -> torch.Tensor:
        if self.use_cache:
            safe = name.replace(".", "_")
            cp = os.path.join(self.cache_dir, f"host_{safe}.pt")
            if os.path.exists(cp):
                return torch.load(cp, map_location="cpu", weights_only=True)
            t = self._raw_torch(name)
            tmp = os.path.join(self.cache_dir, f"host_{safe}.tmp.{os.getpid()}.pt")
            torch.save(t, tmp)
            os.replace(tmp, cp)
            return t
        return self._raw_torch(name)

    def linear(self, name, how=None, dtype=ttnn.bfloat16):
        """Weight [out,in] stored as [1,1,in,out] for x @ W. how: None=replicated, 'col' shards `out`
        (dim 3), 'row' shards `in` (dim 2)."""
        _, _, shape = self.ck.weight_map[name] if isinstance(self.ck.weight_map.get(name), tuple) else (None, None, None)
        if self.use_cache:
            dt_tag = "fp32" if dtype == ttnn.float32 else "bf16"
            safe = f"{name.replace('.', '_')}_{how or 'rep'}_{dt_tag}"
            cp = os.path.join(self.cache_dir, f"{safe}.tensorbin")
            if os.path.exists(cp):
                t = ttnn.load_tensor(cp, device=self.dev)
                self.bytes += os.path.getsize(cp) // 8
                return t
        w = self._raw_torch(name).T.contiguous()
        mapper = {None: rep(self.dev), "col": shard(self.dev, 3), "row": shard(self.dev, 2)}[how]
        self.bytes += w.numel() * 2 // (1 if how is None else 8)
        t = to_dev(self.dev, w, mapper, dtype)
        if self.use_cache:
            tmp = os.path.join(self.cache_dir, f"{safe}.tmp.{os.getpid()}.tensorbin")
            ttnn.dump_tensor(tmp, t)
            os.replace(tmp, cp)
        return t

    def vec(self, name, dtype=ttnn.bfloat16, how=None):
        if self.use_cache:
            dt_tag = "fp32" if dtype == ttnn.float32 else "bf16"
            safe = f"{name.replace('.', '_')}_{how or 'rep'}_{dt_tag}"
            cp = os.path.join(self.cache_dir, f"{safe}.tensorbin")
            if os.path.exists(cp):
                t = ttnn.load_tensor(cp, device=self.dev)
                self.bytes += os.path.getsize(cp) // 8
                return t
        v = self._raw_torch(name)
        self.bytes += v.numel() * 2
        t = to_dev(self.dev, v, shard(self.dev, 3) if how == "col" else None, dtype)
        if self.use_cache:
            tmp = os.path.join(self.cache_dir, f"{safe}.tmp.{os.getpid()}.tensorbin")
            ttnn.dump_tensor(tmp, t)
            os.replace(tmp, cp)
        return t

    def layer(self, L, prefix="layers"):
        p, w = f"{prefix}.{L}", {}
        w["attn_norm"], w["ffn_norm"] = self.vec(f"{p}.attn_norm.weight"), self.vec(f"{p}.ffn_norm.weight")
        for k in ("hc_attn_base", "hc_ffn_base", "hc_attn_scale", "hc_ffn_scale"):
            w[k] = self.torch(f"{p}.{k}")
        a = f"{p}.attn"
        w["wq_a"], w["q_norm"] = self.linear(f"{a}.wq_a.weight"), self.vec(f"{a}.q_norm.weight")
        w["wq_b"] = self.linear(f"{a}.wq_b.weight", "col")
        w["wkv"], w["kv_norm"] = self.linear(f"{a}.wkv.weight"), self.vec(f"{a}.kv_norm.weight")
        w["wo_a"] = self.linear(f"{a}.wo_a.weight", "col")
        w["wo_b"] = self.linear(f"{a}.wo_b.weight", "row")
        w["attn_sink"] = self.torch(f"{a}.attn_sink")
        if self.has(f"{a}.compressor.wkv.weight"):
            w["comp_wkv"], w["comp_norm"] = self.linear(f"{a}.compressor.wkv.weight"), self.vec(f"{a}.compressor.norm.weight")
            if self.has(f"{a}.compressor.wgate.weight"):
                w["comp_wgate"] = self.linear(f"{a}.compressor.wgate.weight")
        if self.has(f"{a}.indexer.wq_b.weight"):
            w["idx_wq_b"] = self.linear(f"{a}.indexer.wq_b.weight", "col")
            w["idx_weights_proj"] = self.linear(f"{a}.indexer.weights_proj.weight", "col")
            if self.has(f"{a}.indexer.wk.weight"):
                w["idx_wk"], w["idx_k_norm"] = self.linear(f"{a}.indexer.wk.weight"), self.vec(f"{a}.indexer.k_norm.weight")
        f = f"{p}.ffn"
        w["gate_w32"] = self.linear(f"{f}.gate.weight", dtype=ttnn.float32)
        w["gate_bias"] = self.torch(f"{f}.gate.bias")
        if self.has(f"{f}.gate.bias_vl"):
            w["gate_bias_vl"] = self.torch(f"{f}.gate.bias_vl")
        w["sh_w1"], w["sh_w3"] = self.linear(f"{f}.shared_experts.w1.weight", "col"), self.linear(f"{f}.shared_experts.w3.weight", "col")
        w["sh_w2"] = self.linear(f"{f}.shared_experts.w2.weight", "row")
        if self.has(f"{p}.engram.wkv.weight"):
            w["eg_wkv"] = self.linear(f"{p}.engram.wkv.weight")
            qk_cache = os.path.join(self.cache_dir, f"host_{p.replace('.', '_')}_eg_qk.pt")
            if self.use_cache and os.path.exists(qk_cache):
                w["eg_qk"] = torch.load(qk_cache, map_location="cpu", weights_only=True)
            else:
                w["eg_qk"] = self._raw_torch(f"{p}.engram.q_weight") * self._raw_torch(f"{p}.engram.k_weight")
                if self.use_cache:
                    tmp = os.path.join(self.cache_dir, f"host_{p.replace('.', '_')}_eg_qk.tmp.{os.getpid()}.pt")
                    torch.save(w["eg_qk"], tmp)
                    os.replace(tmp, qk_cache)
        return w

    def top(self):
        return {"norm": self.vec("norm.weight"), "head": self.linear("head.weight", "col")}
