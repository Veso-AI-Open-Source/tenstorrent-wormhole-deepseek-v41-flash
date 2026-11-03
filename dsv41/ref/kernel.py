"""Plain-PyTorch drop-in for DeepSeek-V4.1-Flash `inference/kernel.py` (TileLang/CUDA).

Same six entry points, same signatures, same numerics contract, no GPU. This directory goes on
sys.path AHEAD of the snapshot's `inference/` dir so the official `model.py` imports this module
unmodified. It exists to make the official model the ORACLE for the pure-TT engine: per-layer
reference activations and the expert route log. It is a test fixture, never a serving path.
"""
import torch

FP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
# nibble -> value, matching convert.py FP4_TABLE: bit 3 is the sign
FP4_TABLE = torch.cat([FP4_GRID, -FP4_GRID])


def _pow2_ceil(x: torch.Tensor) -> torch.Tensor:
    """2**ceil(log2(x)) exactly, via frexp (the reference does it with IEEE bit tricks)."""
    m, e = torch.frexp(x.float())
    return torch.ldexp(torch.ones_like(m), torch.where(m == 0.5, e - 1, e))


def _to_fp4(x: torch.Tensor) -> torch.Tensor:
    """Round to the E2M1 grid, ties to the even code (what a hardware FP4 cast does)."""
    grid = FP4_GRID.to(x.device)
    a = x.abs().clamp(max=6.0)
    hi = torch.searchsorted(grid, a.contiguous()).clamp(max=7)
    lo = (hi - 1).clamp(min=0)
    dlo, dhi = a - grid[lo], grid[hi] - a
    pick_hi = (dhi < dlo) | ((dhi == dlo) & (hi % 2 == 0))
    return torch.where(pick_hi, grid[hi], grid[lo]) * torch.sign(x)


def _blocks(x: torch.Tensor, block: int) -> torch.Tensor:
    return x.float().unflatten(-1, (-1, block))


# DSV41_FAKEQUANT=0 turns OFF every activation fake-quant (fp8 linear inputs, fp8 window KV, fp4
# compressed KV, fp4 indexer q/k) while leaving weights exact. That is the device's v0 semantics
# (bf16 activations), so fixtures dumped in this mode gate the IMPLEMENTATION tightly, and the gap
# between the two modes is measured on its own (stage s02b) instead of polluting every sublayer gate.
import os as _os
FAKEQUANT = _os.environ.get("DSV41_FAKEQUANT", "1") != "0"


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    assert x.size(-1) % block_size == 0
    if not FAKEQUANT:
        return x if inplace else (x.float(), torch.ones(*x.shape[:-1], x.size(-1) // block_size))
    xb = _blocks(x, block_size)
    amax = xb.abs().amax(-1, keepdim=True).clamp(min=1e-4)
    s = _pow2_ceil(amax / 448.0) if scale_fmt is not None else amax / 448.0
    y8 = (xb / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    if inplace:
        x.copy_((y8.float() * s).flatten(-2).to(x.dtype))
        return x
    return y8.flatten(-2), s.squeeze(-1).to(scale_dtype)


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    assert scale_dtype in (torch.float8_e8m0fnu, torch.float8_e4m3fn)
    assert x.size(-1) % block_size == 0
    if not inplace:
        raise NotImplementedError("packed FP4 output is unused by model.py; only inplace is shimmed")
    if not FAKEQUANT:
        return x
    xb = _blocks(x, block_size)
    amax = xb.abs().amax(-1, keepdim=True)
    if scale_dtype == torch.float8_e4m3fn:  # compressed KV: E4M3 scale per 16
        s = (amax.clamp(min=6 * 2.0**-9) / 6.0).to(torch.float8_e4m3fn).float()
    else:  # indexer: E8M0 scale per 32
        s = _pow2_ceil(amax.clamp(min=6 * 2.0**-126) / 6.0)
    x.copy_((_to_fp4(xb / s) * s).flatten(-2).to(x.dtype))
    return x


def _deq_fp8_weight(b: torch.Tensor, b_s: torch.Tensor, block: int) -> torch.Tensor:
    n, k = b.shape
    s = b_s.float().repeat_interleave(block, 0)[:n].repeat_interleave(block, 1)[:, :k]
    return b.float() * s


# Optional hook: fn(W float32 [N,K], packed_weight) -> W, applied to every routed-expert matrix after dequant. The
# quality gate sets it to a bfp4 round trip so the oracle sees exactly the device's weights.
EXPERT_WEIGHT_TRANSFORM = None

try:
    import fp4lib as _fp4lib          # C/AVX-512 path, ~30x faster than the torch LUT below
except Exception:                     # pragma: no cover
    _fp4lib = None


def dequant_fp4_weight(b: torch.Tensor, b_s: torch.Tensor) -> torch.Tensor:
    """[N, K//2] packed E2M1 (low nibble first) + [N, K//32] E8M0 -> [N, K] float32."""
    if _fp4lib is not None and b.dim() == 2 and b.is_contiguous() and b_s.is_contiguous():
        w = _fp4lib.dequant_fp4(b.view(torch.uint8), b_s.view(torch.uint8))
        return EXPERT_WEIGHT_TRANSFORM(w, b) if EXPERT_WEIGHT_TRANSFORM is not None else w   # b identifies the matrix (data_ptr)
    u = b.view(torch.uint8)
    idx = torch.stack([u & 0x0F, u >> 4], dim=-1).flatten(-2).long()
    return FP4_TABLE.to(b.device)[idx] * b_s.float().repeat_interleave(32, dim=-1)


def _deq_act(a: torch.Tensor, a_s: torch.Tensor, block: int) -> torch.Tensor:
    return (_blocks(a, block) * a_s.float().unsqueeze(-1)).flatten(-2)


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size=128):
    out = _deq_act(a, a_s, block_size) @ _deq_fp8_weight(b, b_s, block_size).T
    return out.to(torch.get_default_dtype())


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size=128):
    out = _deq_act(a, a_s, act_block_size) @ dequant_fp4_weight(b, b_s).T
    return out.to(torch.get_default_dtype())


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale, chunk=128):
    """q [b,s,h,d], kv [b,n,d] (K = V, one shared head), topk_idxs [b,s,k] with -1 = masked.
    Softmax runs over the k gathered entries PLUS one learned sink logit per head whose value
    is zero, so a row with no valid index comes out all-zero, as in the reference kernel."""
    b, s, h, d = q.shape
    out = torch.empty_like(q)
    for c0 in range(0, s, chunk):
        idx = topk_idxs[:, c0 : c0 + chunk].long()
        valid = idx >= 0
        g = torch.gather(kv.float().unsqueeze(1).expand(-1, idx.size(1), -1, -1), 2,
                         idx.clamp(min=0).unsqueeze(-1).expand(-1, -1, -1, d))          # [b,c,k,d]
        sc = torch.einsum("bchd,bckd->bchk", q[:, c0 : c0 + chunk].float(), g) * softmax_scale
        sc = sc.masked_fill(~valid.unsqueeze(2), float("-inf"))
        sink = attn_sink.float().view(1, 1, h, 1).expand(b, idx.size(1), h, 1)
        p = torch.cat([sc, sink], dim=-1).softmax(dim=-1)[..., :-1]
        out[:, c0 : c0 + chunk] = torch.einsum("bchk,bckd->bchd", p, g).to(q.dtype)
    return out


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    hc = hc_mult
    m = mixes.float()
    pre = torch.sigmoid(m[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(m[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = (m[..., 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).unflatten(-1, (hc, hc))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb
