"""Engram conditional memory. HOST: hash token n-grams to row ids (integer math on token ids) and fetch
24 FP8 rows per token from the 98 GB table straight out of the checkpoint mmap (NVMe / page cache).
DEVICE: everything else: the 6144 -> 25600 projection, the normalized-dot gate, the residual add.
Reference: engram.py NgramHashState, model.py ParallelEngramEmbedding / Engram."""
import numpy as np
import torch
import ttnn
from mesh import to_dev, mm, rep
from weights import E4M3

HC, DIM = 4, 5120


class EngramHost:
    def __init__(self, ck, tables_npz, layer_ids=(1, 14)):
        t = np.load(tables_npz)
        self.token_map, self.primes, self.offsets = t["token_map"], t["primes"], t["offsets"]
        self.mult, self.pad, self.layer_ids = t["multipliers"], int(t["pad_id"]), list(layer_ids)
        self.tab = {}
        for L in layer_ids:
            w, _, ws = ck.raw(f"layers.{L}.engram.embed.weight"); s, _, ss = ck.raw(f"layers.{L}.engram.embed.scale")
            self.tab[L] = (w.reshape(ws), s.reshape(ss))
        self.hist = []                                   # compressed ids of every token so far

    def hash_ids(self, token_ids, start_pos, token_mask=None):
        """-> [s, n_layers, 24] row ids, identical to NgramHashState.forward. token_mask [s] bool: False = a token
        that takes no part in any n-gram (image spans): it is DEAD, and a look-back stops at it."""
        comp = self.token_map[np.asarray(token_ids)].astype(np.int64)
        if token_mask is not None: comp = np.where(np.asarray(token_mask), comp, -1)
        self.hist = self.hist[:start_pos] + comp.tolist()
        cache = np.asarray(self.hist, dtype=np.int64); n = self.primes.shape[1] + 1
        pos = np.arange(start_pos, start_pos + len(token_ids)); blocked = np.zeros(len(pos), dtype=bool); cols = []
        for sh in range(n):
            src = cache[np.maximum(pos - sh, 0)]
            blocked = blocked | (pos < sh) | (src == -1)
            cols.append(np.where(blocked, self.pad, src))
        toks = np.stack(cols, -1)                                                                                     # [s, n]
        prod = toks[:, None, :] * self.mult[None]                                                                 # [s, layers, n] int64 (wraps like torch)
        rolling, out = prod[..., 0], []
        for i in range(1, n):
            rolling = np.bitwise_xor(rolling, prod[..., i])
            out.append(rolling[..., None] % self.primes[None, :, i - 1])
        return np.concatenate(out, -1) + self.offsets[None]

    def rows(self, L, ids):
        """ids [s,24] -> dequantized rows [s, 24*256] float32 (format decode, not model math)."""
        w, sc = self.tab[L]
        v = E4M3[w[ids]].reshape(*ids.shape, -1, 32) * np.ldexp(np.float32(1), sc[ids].astype(np.int32) - 127)[..., None]
        return torch.from_numpy(v.reshape(ids.shape[0], -1).astype(np.float32))


class Engram:
    def __init__(self, dev, w):
        self.dev, self.wkv = dev, w["eg_wkv"]
        self.qk = to_dev(dev, w["eg_qk"].view(1, HC, 1, DIM))
        self.rows1 = None                                                                 # decode: fixed [1,1,1,6144] rows buffer
        self.rows1_rm = None                                                              # decode: fixed [1,1,1,6144] ROW_MAJOR upload buffer

    def put_rows(self, rows):
        """Decode: write this token's hashed rows [1,6144] into the fixed buffer (outside any trace)."""
        w = rows.numel()
        if self.rows1 is None:
            self.rows1 = to_dev(self.dev, torch.zeros(1, 1, 1, w))
            self.rows1_rm = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, 1, w]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, self.dev, ttnn.DRAM_MEMORY_CONFIG
            )
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(rows.to(torch.bfloat16).view(1, 1, 1, w).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rep(self.dev)),
            self.rows1_rm,
        )
        return self.rows1

    def forward(self, x, rows, token_mask=None):
        """x [1,4,s,d] stream; rows [s,6144] from the host -> x + gate * value, [1,4,s,d]."""
        s = x.shape[2]
        if rows is self.rows1 and self.rows1_rm is not None:
            from rope_cache_op import tilize_row0_1
            tilize_row0_1(self.rows1_rm, self.rows1)
        rows_dev = rows if isinstance(rows, ttnn.Tensor) else to_dev(self.dev, rows.view(1, 1, s, -1))   # decode trace: rows already at a fixed address
        kv = mm(rows_dev, self.wkv)                                                       # [1,1,s,5*d]: four keys then the value
        key = ttnn.concat([kv[:, :, :, i * DIM : (i + 1) * DIM] for i in range(HC)], dim=1)   # [1,4,s,d]
        val = kv[:, :, :, HC * DIM :]                                                     # [1,1,s,d]
        ms = lambda t: ttnn.rsqrt(ttnn.add(ttnn.mean(ttnn.mul(t, t), dim=3, keepdim=True), 1e-20))
        dot = ttnn.sum(ttnn.mul(ttnn.mul(x, self.qk), key), dim=3, keepdim=True)
        dot = ttnn.mul(ttnn.mul(ttnn.mul(dot, ms(x)), ms(key)), DIM ** -0.5)               # [1,4,s,1]
        mag = ttnn.sqrt(ttnn.clamp(ttnn.abs(dot), min=1e-6))
        gate = ttnn.sigmoid(ttnn.mul(mag, ttnn.sign(dot)))
        if token_mask is not None:                                                         # image tokens: gate shut
            gate = ttnn.mul(gate, to_dev(self.dev, token_mask.float().view(1, 1, -1, 1)))
        return ttnn.add(x, ttnn.mul(gate, val))
