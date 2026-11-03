"""Mesh plumbing shared by every device stage. 1x8 ring is the only topology traced CCL works on
here (Kimi-engine finding); two command queues so expert paging (CQ1) overlaps compute (CQ0)."""
import torch
import ttnn

TP = 8

# Wormhole matmul defaults (LoFi, 16-bit dest accumulation) cost ~3% per attention block in bf16.
# HiFi4 + fp32 accumulation is the accuracy-first setting; every matmul in this engine goes through mm().
CK = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)


def mm(a, b, **kw):
    return ttnn.matmul(a, b, compute_kernel_config=CK, **kw)


def open_mesh(trace_region_mb=0, two_cq=True):
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING, ttnn.FabricReliabilityMode.RELAXED_INIT)
    kw = {"num_command_queues": 2} if two_cq else {}
    if trace_region_mb:
        kw["trace_region_size"] = trace_region_mb << 20
    return ttnn.open_mesh_device(ttnn.MeshShape(1, TP), **kw)


def close_mesh(dev):
    ttnn.close_mesh_device(dev)


def rep(dev):
    return ttnn.ReplicateTensorToMesh(dev)


def shard(dev, dim):
    return ttnn.ShardTensorToMesh(dev, dim=dim)


def to_dev(dev, t: torch.Tensor, mapper=None, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    """torch -> mesh. 4D-ify, replicate unless a mapper says otherwise."""
    while t.dim() < 4:
        t = t.unsqueeze(0)
    return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=layout, device=dev, mesh_mapper=mapper or rep(dev))


def chip(t, i=0) -> torch.Tensor:
    """One chip's copy/shard back on the host. Small tensors only: bulk D2H runs at ~0.09 GB/s."""
    # Reads all eight and picks one ON PURPOSE. Reading a single device's tensor is 0.25 ms cheaper, but
    # the first full-model run that did it (2 CQs + generic_op + CCL) spun at 120% CPU for 27 minutes with
    # no progress and had to be killed (board wedged -> `tt-smi -r`). The full read also keeps the eight
    # queues in lockstep once per layer. 10 ms/token is not worth a hang.
    return ttnn.to_torch(ttnn.get_device_tensors(ttnn.from_device(t))[i]).float()


def gather(dev, t, dim) -> torch.Tensor:
    return ttnn.to_torch(ttnn.from_device(t), mesh_composer=ttnn.ConcatMeshToTensor(dev, dim=dim)).float()


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().flatten(), b.float().flatten()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def rms_norm(x, weight, eps=1e-20):
    """x * rsqrt(mean(x^2)) * w from plain ops (or single-launch HiFi4/fp32-acc ttnn.rms_norm at s=1 decode where
    [1,1,1,5120] fits comfortably in L1). TRAP: stock ttnn.rms_norm with default LoFi is WRONG ENOUGH TO MATTER on
    this model's residual stream (massive-activation channels: max 374 vs rms 7): median per-row error
    3.3%, 17% worst, 28% of rows over 5% on the 1300-token fixture; and its fp32-accumulate variant does
    not fit L1 at width 5120 when s=1300."""
    if x.shape[2] <= 32:
        return ttnn.rms_norm(x, weight=weight, epsilon=eps, compute_kernel_config=CK)
    ms = ttnn.mean(ttnn.mul(x, x), dim=3, keepdim=True)
    return ttnn.mul(ttnn.mul(x, ttnn.rsqrt(ttnn.add(ms, eps))), weight)


def blend_row(dev, cache, j, row):
    """cache[..., j, :] = row, exactly, from plain ops (see the update_cache trap): cache [1,1,N,d], row [1,1,1,d]."""
    n = cache.shape[2]
    hot = torch.zeros(1, 1, n, 1); hot[0, 0, j, 0] = 1.0
    return ttnn.add(ttnn.mul(cache, to_dev(dev, 1.0 - hot)), mm(to_dev(dev, hot), row))
