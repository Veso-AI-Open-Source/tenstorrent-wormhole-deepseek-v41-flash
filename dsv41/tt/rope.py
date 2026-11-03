"""RoPE exactly as the reference does it: adjacent pairs are complex numbers, only the LAST 64 dims of
each 512-d head rotate, YaRN on the compressed-KV layers. cos/sin tables are built on the host once
and uploaded per step (positions are scheduling, not model math)."""
import math
import torch
import ttnn
from mesh import to_dev, CK

ROPE = 64


def freqs(seqlen, base, original_seq_len=0, factor=16, beta_fast=32, beta_slow=1, dim=ROPE):
    f = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        cd = lambda r: dim * math.log(original_seq_len / (r * 2 * math.pi)) / (2 * math.log(base))
        low, high = max(math.floor(cd(beta_fast)), 0), min(math.ceil(cd(beta_slow)), dim - 1)
        smooth = 1 - ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        f = f / factor * (1 - smooth) + f * smooth
    return torch.outer(torch.arange(seqlen, dtype=torch.float32), f)          # [seqlen, 32] angles


def tables(dev, angles: torch.Tensor):
    """angles [s,32] -> (cos, sin) each [1,1,s,64] with every value repeated for its pair."""
    c, s = angles.cos().repeat_interleave(2, -1), angles.sin().repeat_interleave(2, -1)
    return to_dev(dev, c.view(1, 1, *c.shape)), to_dev(dev, s.view(1, 1, *s.shape))


_T = {}


def _trans(dev):
    """The 32x32 pair-swap matrix rotary_embedding_llama wants: (x @ T)[2i] = -x[2i+1], (x @ T)[2i+1] = x[2i]."""
    if id(dev) not in _T:
        t = torch.zeros(32, 32)
        for i in range(16):
            t[2 * i + 1, 2 * i] = -1.0; t[2 * i, 2 * i + 1] = 1.0
        _T[id(dev)] = to_dev(dev, t.view(1, 1, 32, 32))
    return _T[id(dev)]


def apply(x, cos, sin, inverse=False):
    """x [1,h,s,d]: rotate the last 64 dims, leave the rest alone. At s<=32 decode/verify, runs in-place on the last 2 tiles
    per head via rope_1 (1 launch vs 4-5 launches, zero slice/concat copies)."""
    if x.shape[2] <= 32:
        from rope_cache_op import rope_1
        return rope_1(x, cos, sin, inverse=inverse)
    d = x.shape[3]
    # HiFi4 is NOT optional here: with the op's default fidelity the sublayer gates did not move (2-3%) but
    # the 512-token gate lost 4 decisive flips and decode KL went 0.014 -> 0.052. Rotations touch every key.
    rot = ttnn.experimental.rotary_embedding_llama(x[:, :, :, d - ROPE :], cos, ttnn.neg(sin) if inverse else sin,
                                                   _trans(x.device()), is_decode_mode=False, compute_kernel_config=CK)
    return ttnn.concat([x[:, :, :, : d - ROPE], rot], dim=3)
