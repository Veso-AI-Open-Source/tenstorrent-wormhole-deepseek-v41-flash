"""Host expert store = a directory of pre-packed .tensorbin files, mmapped by ttnn.load_tensor.
No copies, no LRU: the kernel page cache is the cache. Missing experts are packed on-demand via
libfp4.so and atomically cached to disk so any prompt works out-of-the-box."""
import concurrent.futures
import json
import os
import sys
import threading
import numpy as np
import torch
import ttnn


class ExpertStore:
    def __init__(self, fmt="mixed", root=None):
        """fmt "mixed" = the final store weights/experts/ (per-layer format in manifest.json);
        "bfp8"/"bfp4" = a single-format directory used by the implementation gates."""
        self.root = root or os.environ.get("DSV41_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        base = os.path.join(self.root, "weights")
        self.dir = os.path.join(base, "experts" if fmt == "mixed" else f"experts_{fmt}")
        os.makedirs(self.dir, exist_ok=True)
        if fmt == "mixed":
            mp = f"{self.dir}/manifest.json"
            if not os.path.exists(mp):
                manifest = {
                    **{f"L{L:02d}": ("bfp8" if L < 20 else "bfp4") for L in range(40)},
                    **{f"Lm{m}": "bfp4" for m in range(3)},
                }
                tmp_mp = f"{mp}.tmp.{os.getpid()}"
                with open(tmp_mp, "w") as f:
                    json.dump(manifest, f, indent=1)
                os.replace(tmp_mp, mp)
            self.manifest = json.load(open(mp))
        else:
            self.manifest = None
        from collections import OrderedDict
        self.live = OrderedDict()
        self.max_live = int(os.environ.get("MK_LIVE_EXPERTS_MAX", "8192"))
        self._ck = None
        self._hdr = None
        self._init_lock = threading.Lock()
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    def fmt(self, layer):
        return self.manifest[f"L{layer:02d}" if isinstance(layer, int) else layer] if self.manifest else os.path.basename(self.dir).split("_")[1]

    def _init_ck(self):
        if self._ck is not None:
            return
        with self._init_lock:
            if self._ck is not None:
                return
            ref_dir = os.path.join(self.root, "ref")
            if ref_dir not in sys.path:
                sys.path.insert(0, ref_dir)
            if self.root not in sys.path:
                sys.path.insert(0, self.root)
            from ckpt import Checkpoint
            self._ck = Checkpoint()
            fmtdir = os.path.join(self.root, "dumps", "fmt")
            self._hdr = {
                f: {p: open(f"{fmtdir}/header_{p}_{f}.bin", "rb").read() for p in ("w1", "w2")}
                for f in ("bfp4", "bfp8")
            }

    def _pack_one_proj(self, layer, e, pj):
        lname = f"L{layer:02d}" if isinstance(layer, int) else layer
        d = f"{self.dir}/{lname}"
        out_p = f"{d}/e{e:03d}.{pj}.tensorbin"
        if os.path.exists(out_p):
            return
        os.makedirs(d, exist_ok=True)
        self._init_ck()
        import fp4lib
        f = self.fmt(layer)
        bits = 3 if f == "bfp4" else 7
        prefix = "mtp" if (isinstance(layer, str) and layer.startswith("Lm")) else "layers"
        lidx = int(layer[2:]) if (isinstance(layer, str) and layer.startswith("Lm")) else (int(layer[1:]) if isinstance(layer, str) else int(layer))
        hdr_b = self._hdr[f]["w2" if pj == "w2" else "w1"]
        q, _, _ = self._ck.raw(f"{prefix}.{lidx}.ffn.experts.{e}.{pj}.weight")
        s, _, _ = self._ck.raw(f"{prefix}.{lidx}.ffn.experts.{e}.{pj}.scale")
        pay_bytes = 8 * 1440 * (64 + (1024 if bits == 7 else 512))
        buf = np.empty(len(hdr_b) + pay_bytes, dtype=np.uint8)
        buf[:len(hdr_b)] = np.frombuffer(hdr_b, dtype=np.uint8)
        fp4lib.pack_expert_shards_into(q, s, pj == "w2", bits, buf, len(hdr_b))
        tmp_p = f"{out_p}.tmp.{os.getpid()}.{threading.get_ident()}"
        with open(tmp_p, "wb") as fp:
            fp.write(buf)
        os.replace(tmp_p, out_p)

    def ensure_packed_many(self, keys):
        futs = []
        for layer, e in keys:
            e = int(e)
            lname = f"L{layer:02d}" if isinstance(layer, int) else layer
            d = f"{self.dir}/{lname}"
            for pj in ("w1", "w2", "w3"):
                if not os.path.exists(f"{d}/e{e:03d}.{pj}.tensorbin"):
                    futs.append(self._pool.submit(self._pack_one_proj, layer, e, pj))
        for fut in futs:
            fut.result()

    def _ensure_packed(self, layer, e):
        lname = f"L{layer:02d}" if isinstance(layer, int) else layer
        d = f"{self.dir}/{lname}"
        if os.path.exists(f"{d}/e{e:03d}.w3.tensorbin") and os.path.exists(f"{d}/e{e:03d}.w1.tensorbin") and os.path.exists(f"{d}/e{e:03d}.w2.tensorbin"):
            return d
        self.ensure_packed_many([(layer, e)])
        return d

    def get(self, layer, e):
        k = (layer, int(e))
        if k in self.live:
            self.live.move_to_end(k)
            return self.live[k]
        d = self._ensure_packed(layer, int(e))
        self.live[k] = {pj: ttnn.load_tensor(f"{d}/e{int(e):03d}.{pj}.tensorbin") for pj in ("w1", "w2", "w3")}
        while len(self.live) > self.max_live:
            self.live.popitem(last=False)
        return self.live[k]
