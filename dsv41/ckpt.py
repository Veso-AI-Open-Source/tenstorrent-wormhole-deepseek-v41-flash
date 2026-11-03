"""Dependency-free reader for the DeepSeek-V4.1-Flash HF checkpoint (numpy only).

safetensors/torch in the tt-metal container predate F8_E8M0 and packed-FP4 dtypes, so tensors are
read as raw bytes straight from the shard via the safetensors header and decoded here.
Experts: `layers.L.ffn.experts.E.w{1,2,3}.weight` int8 [out, in/2] = two E2M1 nibbles per byte, LOW
nibble first; `.scale` E8M0 [out, in/32] (value = 2**(byte-127)). w1=gate, w3=up, w2=down.
"""
import glob, json, mmap, os, struct, threading
import numpy as np

SNAP = sorted(glob.glob(os.path.expanduser(os.environ.get(
    "DSV41_SNAP", "~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/*"))))[-1]
FP4_TABLE = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)


class Checkpoint:
    def __init__(self, snap=SNAP):
        self.snap = snap
        self.weight_map = json.load(open(f"{snap}/model.safetensors.index.json"))["weight_map"]
        self._shards = {}
        self._lock = threading.Lock()

    def _shard(self, fname):
        s = self._shards.get(fname)
        if s is not None:
            return s
        with self._lock:
            if fname not in self._shards:
                f = open(f"{self.snap}/{fname}", "rb")
                n = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(n)); hdr.pop("__metadata__", None)
                self._shards[fname] = (mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ), 8 + n, hdr)
            return self._shards[fname]

    def raw(self, name):
        """(uint8 view of the tensor bytes, dtype string, shape). Zero-copy."""
        mm, base, hdr = self._shard(self.weight_map[name])
        meta = hdr[name]; a, b = meta["data_offsets"]
        return np.frombuffer(mm, dtype=np.uint8, count=b - a, offset=base + a), meta["dtype"], meta["shape"]

    def expert_proj(self, layer, expert, proj, prefix="layers"):
        """Dequantized float32 [out, in] for proj in {w1,w2,w3}, plus the raw E8M0 exponent bytes."""
        stem = f"{prefix}.{layer}.ffn.experts.{expert}.{proj}"
        q, qd, qs = self.raw(stem + ".weight"); s, sd, ss = self.raw(stem + ".scale")
        assert qd == "I8" and sd == "F8_E8M0", (qd, sd)
        out, half = qs; assert ss == [out, half * 2 // 32], (qs, ss)
        q = q.reshape(out, half)
        vals = FP4_TABLE[np.stack([q & 0x0F, q >> 4], axis=-1).reshape(out, half * 2)]
        scale = np.ldexp(np.float32(1.0), s.reshape(ss).astype(np.int32) - 127)
        return vals * np.repeat(scale, 32, axis=1), s.reshape(ss)
