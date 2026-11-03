"""DeepSeek-ViT + aligner on the mesh (bf16, replicated; stock ops). 32 pre-norm blocks, 16 heads x 64, 2D RoPE in
the half-split convention, SwiGLU MLP, full bidirectional attention over one image. The 3x3 pixel-unshuffle
between ViT and aligner is a pure rearrangement and is done on the host. Reference: inference/vision.py."""
import torch
import torch.nn.functional as F
import ttnn
from mesh import to_dev, mm, chip, rms_norm

DIM, HEADS, HD, LAYERS, RATIO = 1024, 16, 64, 32, 3
import os as _os
SOFTMAX32 = _os.environ.get("DSV41_VIT_SOFTMAX32", "1") != "0"
NORM32 = _os.environ.get("DSV41_VIT_NORM32", "1") != "0"


def _norm(x, w, eps=1e-6):
    """RMSNorm; the reference does the statistics in fp32 (x.float()) and rounds once at the end."""
    if not NORM32:
        return rms_norm(x, w, eps=eps)
    x32 = ttnn.typecast(x, ttnn.float32)
    ms = ttnn.mean(ttnn.mul(x32, x32), dim=3, keepdim=True)
    return ttnn.typecast(ttnn.mul(ttnn.mul(x32, ttnn.rsqrt(ttnn.add(ms, eps))), w), ttnn.bfloat16)      # w is fp32: one rounding


class Vision:
    def __init__(self, dev, R):
        self.dev = dev
        W = lambda n: to_dev(dev, R.torch(n).T.contiguous())                       # [in, out]
        V = lambda n: to_dev(dev, R.torch(n).view(1, 1, 1, -1))
        N = lambda n: to_dev(dev, R.torch(n).view(1, 1, 1, -1), dtype=ttnn.float32 if NORM32 else ttnn.bfloat16)   # norm weights
        self.pe_w, self.pe_b = W("vision.patch_embed.proj.weight"), V("vision.patch_embed.proj.bias")
        self.blocks = [{"n1": N(f"vision.blocks.{i}.norm1.weight"), "n2": N(f"vision.blocks.{i}.norm2.weight"),
                        "qkv_w": W(f"vision.blocks.{i}.attn.wqkv.weight"), "qkv_b": V(f"vision.blocks.{i}.attn.wqkv.bias"),
                        "wo_w": W(f"vision.blocks.{i}.attn.wo.weight"), "wo_b": V(f"vision.blocks.{i}.attn.wo.bias"),
                        "w1": W(f"vision.blocks.{i}.mlp.w1.weight"), "w2": W(f"vision.blocks.{i}.mlp.w2.weight")} for i in range(LAYERS)]
        self.norm = N("vision.norm.weight")
        self.a1_w, self.a1_b, self.a2_w, self.a2_b = W("aligner.w1.weight"), V("aligner.w1.bias"), W("aligner.w2.weight"), V("aligner.w2.bias")

    @staticmethod
    def _rot(x, cos, sin):
        """half-split rotation over head_dim 64: [x1*cos - x2*sin, x2*cos + x1*sin]. x [1,16,n,64], cos/sin [1,1,n,32].
        Done in fp32 like the reference (x.float(), one rounding at the end)."""
        if NORM32: x = ttnn.typecast(x, ttnn.float32)
        x1, x2 = x[:, :, :, : HD // 2], x[:, :, :, HD // 2 :]
        r = ttnn.concat([ttnn.sub(ttnn.mul(x1, cos), ttnn.mul(x2, sin)), ttnn.add(ttnn.mul(x2, cos), ttnn.mul(x1, sin))], dim=3)
        return ttnn.typecast(r, ttnn.bfloat16) if NORM32 else r

    def vit(self, patches: torch.Tensor, n_h: int, n_w: int):
        n = patches.shape[0]
        inv = 1.0 / (10000.0 ** (torch.arange(0, HD // 2, 2, dtype=torch.float32) / (HD // 2)))
        hp, wp = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w), torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
        fr = (torch.stack([hp, wp], -1).reshape(-1, 2, 1).float() * inv).flatten(1)                     # [n, 32]
        cdt = ttnn.float32 if NORM32 else ttnn.bfloat16
        cos, sin = to_dev(self.dev, fr.cos().view(1, 1, n, -1), dtype=cdt), to_dev(self.dev, fr.sin().view(1, 1, n, -1), dtype=cdt)
        x = ttnn.add(mm(to_dev(self.dev, patches.flatten(1).view(1, 1, n, -1)), self.pe_w), self.pe_b)  # [1,1,n,1024]
        for b in self.blocks:
            h = ttnn.add(mm(_norm(x, b["n1"]), b["qkv_w"]), b["qkv_b"])                   # [1,1,n,3072]
            q, k, v = (ttnn.transpose(ttnn.reshape(h[:, :, :, i * DIM : (i + 1) * DIM], (1, n, HEADS, HD)), 1, 2) for i in range(3))
            q, k = self._rot(q, cos, sin), self._rot(k, cos, sin)
            # Scores stay fp32 end to end (the reference's CPU flash kernel never rounds them to bf16: a score of 20
            # rounded to bf16 is off by 0.06 = 6% on that weight). Softmax in fp32, probabilities to bf16 for p @ v.
            if SOFTMAX32:
                sc = ttnn.mul(mm(q, ttnn.transpose(k, 2, 3), dtype=ttnn.float32), HD ** -0.5)             # [1,16,n,n] fp32
                p = ttnn.typecast(ttnn.softmax(sc, dim=-1), ttnn.bfloat16)
            else:
                p = ttnn.softmax(ttnn.mul(mm(q, ttnn.transpose(k, 2, 3)), HD ** -0.5), dim=-1)
            o = ttnn.reshape(ttnn.transpose(mm(p, v), 1, 2), (1, 1, n, DIM))
            x = ttnn.add(x, ttnn.add(mm(o, b["wo_w"]), b["wo_b"]))
            gu = mm(_norm(x, b["n2"]), b["w1"])                                           # [1,1,n,5632]: gate | up
            half = gu.shape[3] // 2
            x = ttnn.add(x, mm(ttnn.mul(ttnn.silu(gu[:, :, :, :half]), gu[:, :, :, half:]), b["w2"]))
        return _norm(x, self.norm)

    def encode(self, patches, n_h, n_w):
        """-> (vit_out [n,1024], image embeddings [m,5120]) on the HOST (they are merged into embedding rows there)."""
        vit = chip(self.vit(patches, n_h, n_w))[0, 0]
        g = vit.view(n_h, n_w, -1).permute(2, 0, 1)
        g = F.pad(g, (0, -n_w % RATIO, 0, -n_h % RATIO))
        u = F.unfold(g.unsqueeze(0), RATIO, stride=RATIO).squeeze(0).transpose(0, 1).contiguous()       # [m, 9216]: a rearrangement, no math
        m_ = u.shape[0]
        e = ttnn.add(mm(ttnn.gelu(ttnn.add(mm(to_dev(self.dev, u.view(1, 1, m_, -1)), self.a1_w), self.a1_b)), self.a2_w), self.a2_b)
        return vit, chip(e)[0, 0]
