"""Manifold-constrained hyper-connections on the mesh (stock ops + one custom kernel in decode).

LAYOUT: the residual stream is x [1, 4, s, d], and the hc copies live in dim 1. NOT [1, s, 4, d]: TILE layout
pads the last two dims to multiples of 32, so a trailing-but-one dim of 4 becomes 32 and every stream tensor
(and every op on it) is 8x too big. Measured: 1.34 GB instead of 168 MB at s=4096.

Each sublayer computes three coefficient sets from the stream (`mixes`): `pre` [1,4,s,1] collapses 4->1 for
the NEXT sublayer's input, `post` [1,4,s,1] and `comb` [1,4(i),s,4(j)] expand this sublayer's output back to
4 and mix the residual through a doubly-stochastic 4x4 (20 Sinkhorn steps).
Reference: model.py Block.hc_mixes / hc_pre / hc_post and kernel.py hc_split_sinkhorn.
"""
import torch
import ttnn
from mesh import mm, to_dev

HC = 4


class MHC:
    def __init__(self, dev, hc_fn_host: torch.Tensor, hc_base: torch.Tensor, hc_scale: torch.Tensor, norm_eps=1e-20, hc_eps=1e-6, iters=20, sink_op=None):
        """hc_fn_host: the checkpoint's [24, 4*d] projection (float32, host)."""
        self.dev, self.eps, self.hc_eps, self.iters, self.sink_op = dev, norm_eps, hc_eps, iters, sink_op
        d = hc_fn_host.shape[1] // HC
        # flat @ fn.T == sum_i x_i @ fn_i.T with fn_i = fn[:, i*d:(i+1)*d]: one batched matmul over the hc dim
        self.fn4 = to_dev(dev, hc_fn_host.float().view(-1, HC, d).permute(1, 2, 0).contiguous().view(1, HC, d, -1))
        s, b = hc_scale.float(), hc_base.float()
        self.scale = to_dev(dev, torch.cat([s[0].repeat(HC), s[1].repeat(HC), s[2].repeat(HC * HC)]).view(1, 1, 1, -1))
        self.base = to_dev(dev, b.view(1, 1, 1, -1))

    def mixes(self, x, ms=None):
        """x [1,4,s,d] -> pre [1,4,s,1], post [1,4,s,1], comb [1,4(i),s,4(j)] (comb[i,j] as in the reference)."""
        s = x.shape[2]
        if (s == 1 or (s <= 8 and ms is not None)) and self.sink_op is not None and hasattr(self.sink_op, "mixes_1"):
            if ms is None:
                ms = ttnn.mean(ttnn.mul(x, x), dim=[1, 3], keepdim=True)
            return self.sink_op.mixes_1(mm(x, self.fn4), ms, self.scale, self.base)
        if ms is None:
            ms = ttnn.mean(ttnn.mul(x, x), dim=[1, 3], keepdim=True)                                     # [1,1,s,1]: one statistic per token over all 4*d values
        m = ttnn.mul(ttnn.sum(mm(x, self.fn4), dim=1, keepdim=True), ttnn.rsqrt(ttnn.add(ms, self.eps)))   # [1,1,s,24]
        m = ttnn.add(ttnn.mul(m, self.scale), self.base)
        col = lambda t: ttnn.transpose(ttnn.reshape(t, (1, s, HC, 1)), 1, 2) if s > 1 else ttnn.reshape(t, (1, HC, 1, 1))   # [1,1,s,4] -> [1,4,s,1] (tiny)
        pre = col(ttnn.add(ttnn.sigmoid(m[:, :, :, 0:HC]), self.hc_eps))
        post = col(ttnn.mul(ttnn.sigmoid(m[:, :, :, HC : 2 * HC]), 2.0))
        comb = ttnn.reshape(m[:, :, :, 2 * HC :], (1, s, HC, HC))
        comb = ttnn.add(ttnn.softmax(comb, dim=-1), self.hc_eps)                                    # rows i, softmax over j
        if s == 1 and self.sink_op is not None:
            return pre, post, ttnn.reshape(self.sink_op(ttnn.reshape(comb, (1, 1, HC, HC))), (1, HC, 1, HC))
        if s <= 32 and self.sink_op is not None:                  # short passes (DSpark verify): one kernel launch for s matrices
            return pre, post, ttnn.transpose(self.sink_op(comb), 1, 2)
        comb = ttnn.transpose(comb, 1, 2) if s > 1 else ttnn.reshape(comb, (1, HC, 1, HC))          # [1,4(i),s,4(j)]
        # KEEP the reference's "+eps" on every sum: near-one-hot softmax rows give column sums ~4e-6, where eps is
        # 25% of the divisor. Dropping it cost 3 decisive flips end to end while the sublayer gate did not move.
        comb = ttnn.div(comb, ttnn.add(ttnn.sum(comb, dim=1, keepdim=True), self.hc_eps))           # columns: over i
        for _ in range(self.iters - 1):
            comb = ttnn.div(comb, ttnn.add(ttnn.sum(comb, dim=3, keepdim=True), self.hc_eps))       # rows: over j
            comb = ttnn.div(comb, ttnn.add(ttnn.sum(comb, dim=1, keepdim=True), self.hc_eps))
        return pre, post, comb

    @staticmethod
    def pre(x, pre_mix):
        """x [1,4,s,d], pre_mix [1,4,s,1] -> [1,1,s,d]: weighted collapse of the copies."""
        return ttnn.sum(ttnn.mul(x, pre_mix), dim=1, keepdim=True)

    @staticmethod
    def post(y, residual, post, comb):
        """y [1,1,s,d], residual [1,4,s,d] -> [1,4,s,d]:  out_j = post_j * y + sum_i comb[i,j] * residual_i."""
        if residual.shape[2] == 1:     # decode: one 4x4 mixing of four d-vectors IS a matmul; 3 ops instead of 15
            d = residual.shape[3]
            ct = comb if comb.shape[1] == 1 else ttnn.transpose(ttnn.reshape(comb, (1, 1, HC, HC)), 2, 3)
            mixed = mm(ct, ttnn.reshape(residual, (1, 1, HC, d)))
            return ttnn.add(ttnn.mul(post, y), ttnn.reshape(mixed, (1, HC, 1, d)))
        mixed = ttnn.concat([ttnn.sum(ttnn.mul(residual, comb[:, :, :, j : j + 1]), dim=1, keepdim=True) for j in range(HC)], dim=1)
        return ttnn.add(ttnn.mul(post, y), mixed)
