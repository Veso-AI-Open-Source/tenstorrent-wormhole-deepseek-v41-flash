"""Fast s=1 decode wrappers for kernels/rope_cache.cpp:
  - rope_1(x, cos, sin, inverse=False): in-place RoPE rotation of last 64 dims of x [1, H, 1, D] (1 launch vs 4-5).
  - rope_reshape_q1(q, cos, sin) / rope_reshape_o1(pv, cos, sin): fused reshape ([1,1,1,4096] <-> [1,8,1,512]) + RoPE.
  - cache_write_1(cache, row, pos_buf, slot_idx, cos=None, sin=None): in-place single-row write with optional fused RoPE.
  - attn_softmax_1(sc_raw, sink1, pos_buf, mode_slot): in-place scale + mask + sink + softmax on [1, 8, 1, k].
  - tilize_row0_1(rm_src, tile_dst, pre_dst=None): 64-worker tilizer from ROW_MAJOR [1,1,1,D] -> TILE [1,HC,1,D].
"""
import os
import ttnn

_HERE = os.path.dirname(os.path.abspath(__file__))
KERNEL = os.path.join(os.path.dirname(_HERE), "kernels", "rope_cache.cpp")

PAGE = 2048
_GRID_CACHE = {}
_DEV_BUFS = {}


def _grid(n_cores: int):
    if n_cores not in _GRID_CACHE:
        if n_cores <= 8:
            crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(n_cores - 1, 0))])
            coords = [(x, 0) for x in range(n_cores)]
        else:
            rows = n_cores // 8
            crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, rows - 1))])
            coords = [(x, y) for y in range(rows) for x in range(8)]
        cb6 = [
            ttnn.CBDescriptor(
                total_size=6 * PAGE,
                core_ranges=crs,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=PAGE)],
            )
        ]
        cb9_both = [
            ttnn.CBDescriptor(
                total_size=9 * PAGE,
                core_ranges=crs,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=PAGE)],
            )
            for i in (0, 1)
        ]
        cb_sm = [
            ttnn.CBDescriptor(
                total_size=38 * PAGE,
                core_ranges=crs,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=PAGE)],
            ),
            ttnn.CBDescriptor(
                total_size=2 * PAGE,
                core_ranges=crs,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=1, data_format=ttnn.bfloat16, page_size=PAGE)],
            ),
        ]
        _GRID_CACHE[n_cores] = (crs, coords, cb6, cb9_both, cb_sm)
    return _GRID_CACHE[n_cores]


def _dev_bufs(dev, s: int = 1):
    did = id(dev) if s == 1 else (id(dev), int(s))
    if did not in _DEV_BUFS:
        _DEV_BUFS[did] = (
            ttnn.allocate_tensor_on_device(ttnn.Shape([1, 8, s, 512]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG),
            ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, s, 4096]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG),
            ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, s, 16160]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)
            if s <= 8
            else None,
        )
    return _DEV_BUFS[did]


def rope_1(x, cos, sin, inverse: bool = False):
    """In-place s<=32 RoPE on the last 64 dims (2 tiles) of each head in x [1, H, s, D]."""
    h = int(x.shape[1])
    s_rows = int(x.shape[2])
    wt = int(x.shape[3]) // 32
    crs, coords, cb6, _, _ = _grid(h)
    inv_s = (1 if inverse else 0) | (s_rows << 1)
    base_args = [int(x.buffer_address()), int(cos.buffer_address()), int(sin.buffer_address()), wt, inv_s]
    rt = ttnn.RuntimeArgs()
    for cid, (cx, cy) in enumerate(coords):
        rt[cx][cy] = base_args + [cid]
    ct = (
        list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
        + list(ttnn.TensorAccessorArgs(cos).get_compile_time_args())
        + list(ttnn.TensorAccessorArgs(sin).get_compile_time_args())
    )
    prog = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=crs,
                compile_time_args=ct,
                runtime_args=rt,
                config=ttnn.ReaderConfigDescriptor(),
            )
        ],
        semaphores=[],
        cbs=cb6,
    )
    ttnn.generic_op([x, cos, sin], prog)
    return x


def _rope_copy(src, cos, sin, dst, inverse: bool):
    crs, coords, _, cb9_both, _ = _grid(32)
    s_rows = int(src.shape[2])
    inv_s = (1 if inverse else 0) | (s_rows << 1)
    base_args = [
        int(src.buffer_address()),
        int(cos.buffer_address()),
        int(sin.buffer_address()),
        int(dst.buffer_address()),
        inv_s,
    ]
    rt = ttnn.RuntimeArgs()
    for cid, (cx, cy) in enumerate(coords):
        rt[cx][cy] = base_args + [cid]
    ct = (
        list(ttnn.TensorAccessorArgs(src).get_compile_time_args())
        + list(ttnn.TensorAccessorArgs(cos).get_compile_time_args())
        + list(ttnn.TensorAccessorArgs(sin).get_compile_time_args())
        + list(ttnn.TensorAccessorArgs(dst).get_compile_time_args())
    )
    prog = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=crs,
                compile_time_args=ct,
                defines=[("MODE_ROPE_COPY", "1")],
                runtime_args=rt,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=crs,
                compile_time_args=ct,
                defines=[("MODE_ROPE_COPY", "1"), ("IS_WRITER", "1")],
                runtime_args=rt,
                config=ttnn.WriterConfigDescriptor(),
            ),
        ],
        semaphores=[],
        cbs=cb9_both,
    )
    ttnn.generic_op([src, cos, sin, dst], prog)
    return dst


def rope_reshape_q1(dev, q, cos, sin):
    """Fused s<=32 reshape([1,1,s,4096] -> [1,8,s,512]) + forward RoPE."""
    q8_buf, _, _ = _dev_bufs(dev, int(q.shape[2]))
    return _rope_copy(q, cos, sin, q8_buf, False)


def rope_reshape_o1(dev, pv, cos, sin):
    """Fused s<=32 inverse RoPE + reshape([1,8,s,512] -> [1,1,s,4096])."""
    _, o1_buf, _ = _dev_bufs(dev, int(pv.shape[2]))
    return _rope_copy(pv, cos, sin, o1_buf, True)


def tilize_row0_1(rm_src, tile_dst, pre_dst=None):
    """Fused 64-worker tilizer from 1-page ROW_MAJOR [1,1,1,D] -> TILE [1,HC,1,D] (+ optional pre0)."""
    crs, coords, _, cb9_both, _ = _grid(32)
    n_tiles = int(tile_dst.shape[3]) // 32
    hc = int(tile_dst.shape[1])
    pre_addr = int(pre_dst.buffer_address()) if pre_dst is not None else 0
    base_args = [
        int(rm_src.buffer_address()),
        int(tile_dst.buffer_address()),
        pre_addr,
        n_tiles,
        hc,
    ]
    rt = ttnn.RuntimeArgs()
    for cid, (cx, cy) in enumerate(coords):
        rt[cx][cy] = base_args + [cid]
    ct = (
        list(ttnn.TensorAccessorArgs(rm_src).get_compile_time_args())
        + list(ttnn.TensorAccessorArgs(tile_dst).get_compile_time_args())
    )
    tensors = [rm_src, tile_dst]
    if pre_dst is not None:
        tensors.append(pre_dst)
    prog = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=crs,
                compile_time_args=ct,
                defines=[("MODE_TILIZE_ROW0", "1")],
                runtime_args=rt,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=crs,
                compile_time_args=ct,
                defines=[("MODE_TILIZE_ROW0", "1"), ("IS_WRITER", "1")],
                runtime_args=rt,
                config=ttnn.WriterConfigDescriptor(),
            ),
        ],
        semaphores=[],
        cbs=cb9_both,
    )
    ttnn.generic_op(tensors, prog)
    return tile_dst


def untilize_head_1(dev, lg_tile, s_rows: int = None):
    """Fused 64-worker untilizer of rows 0..s_rows-1 from TILE [1,1,s,16160] -> ROW_MAJOR [1,1,s_rows,16160]."""
    if s_rows is None:
        s_rows = int(lg_tile.shape[2])
    _, _, lg_rm = _dev_bufs(dev, s_rows)
    crs, coords, _, cb9_both, _ = _grid(32)
    n_tiles = int(lg_tile.shape[3]) // 32
    nt_pack = n_tiles | (int(s_rows) << 16)
    base_args = [
        int(lg_tile.buffer_address()),
        int(lg_rm.buffer_address()),
        nt_pack,
    ]
    rt = ttnn.RuntimeArgs()
    for cid, (cx, cy) in enumerate(coords):
        rt[cx][cy] = base_args + [cid]
    ct = (
        list(ttnn.TensorAccessorArgs(lg_tile).get_compile_time_args())
        + list(ttnn.TensorAccessorArgs(lg_rm).get_compile_time_args())
    )
    prog = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=crs,
                compile_time_args=ct,
                defines=[("MODE_UNTILIZE_ROW0", "1")],
                runtime_args=rt,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=crs,
                compile_time_args=ct,
                defines=[("MODE_UNTILIZE_ROW0", "1"), ("IS_WRITER", "1")],
                runtime_args=rt,
                config=ttnn.WriterConfigDescriptor(),
            ),
        ],
        semaphores=[],
        cbs=cb9_both,
    )
    ttnn.generic_op([lg_tile, lg_rm], prog)
    return lg_rm


def cache_write_1(cache, row, pos_buf, slot_idx: int, cos=None, sin=None):
    """In-place s=1 cache row write: cache[0, 0, pos_buf[slot_idx], :] = row[0, 0, 0, :], with optional fused RoPE."""
    max_rows = int(cache.shape[2])
    wt = int(cache.shape[3]) // 32
    crs, coords, cb6, _, _ = _grid(wt)
    cos_addr = int(cos.buffer_address()) if cos is not None else 0
    sin_addr = int(sin.buffer_address()) if sin is not None else 0
    base_args = [
        int(cache.buffer_address()),
        int(row.buffer_address()),
        int(pos_buf.buffer_address()),
        int(slot_idx),
        max_rows,
        wt,
        cos_addr,
        sin_addr,
    ]
    rt = ttnn.RuntimeArgs()
    for cid, (cx, cy) in enumerate(coords):
        rt[cx][cy] = base_args + [cid]
    ct = (
        list(ttnn.TensorAccessorArgs(cache).get_compile_time_args())
        + list(ttnn.TensorAccessorArgs(row).get_compile_time_args())
        + list(ttnn.TensorAccessorArgs(pos_buf).get_compile_time_args())
    )
    tensors = [cache, row, pos_buf]
    if cos is not None and sin is not None:
        tensors.extend([cos, sin])
    prog = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=crs,
                compile_time_args=ct,
                defines=[("MODE_CACHE_WRITE", "1")],
                runtime_args=rt,
                config=ttnn.ReaderConfigDescriptor(),
            )
        ],
        semaphores=[],
        cbs=cb6,
    )
    ttnn.generic_op(tensors, prog)
    return cache


def cache_write_multi(cache, row, start_idx: int, s_rows: int = 1, is_ring: bool = False, cos=None, sin=None):
    """In-place multi-row cache write: cache[0, 0, j, :] = row[0, 0, r, :] for r in 0..s_rows-1, with optional fused RoPE."""
    if s_rows <= 0:
        return cache
    max_rows = int(cache.shape[2])
    wt = int(cache.shape[3]) // 32
    crs, coords, cb6, _, _ = _grid(wt)
    cos_addr = int(cos.buffer_address()) if cos is not None else 0
    sin_addr = int(sin.buffer_address()) if sin is not None else 0
    packed = (int(start_idx) & 0xFFFF) | ((int(s_rows) & 0xFF) << 16) | ((1 if is_ring else 0) << 24)
    base_args = [
        int(cache.buffer_address()),
        int(row.buffer_address()),
        0,
        packed,
        max_rows,
        wt,
        cos_addr,
        sin_addr,
    ]
    rt = ttnn.RuntimeArgs()
    for cid, (cx, cy) in enumerate(coords):
        rt[cx][cy] = base_args + [cid]
    row_ct = list(ttnn.TensorAccessorArgs(row).get_compile_time_args())
    ct = list(ttnn.TensorAccessorArgs(cache).get_compile_time_args()) + row_ct + row_ct
    tensors = [cache, row]
    if cos is not None and sin is not None:
        tensors.extend([cos, sin])
    prog = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=crs,
                compile_time_args=ct,
                defines=[("MODE_CACHE_WRITE", "1")],
                runtime_args=rt,
                config=ttnn.ReaderConfigDescriptor(),
            )
        ],
        semaphores=[],
        cbs=cb6,
    )
    ttnn.generic_op(tensors, prog)
    return cache


def attn_softmax_1(sc_raw, sink1, pos_buf, mode_slot: int):
    """Fused s=1 scale (512^-0.5) + causal/reach mask + sink logit + softmax in-place on sc_raw [1, 8, 1, k]."""
    kt = int(sc_raw.shape[3]) // 32
    crs, coords, _, _, cb_sm = _grid(8)
    base_args = [
        int(sc_raw.buffer_address()),
        int(sink1.buffer_address()),
        int(pos_buf.buffer_address()),
        kt,
        int(mode_slot),
    ]
    rt = ttnn.RuntimeArgs()
    for cid, (cx, cy) in enumerate(coords):
        rt[cx][cy] = base_args + [cid]
    ct = (
        list(ttnn.TensorAccessorArgs(sc_raw).get_compile_time_args())
        + list(ttnn.TensorAccessorArgs(sink1).get_compile_time_args())
        + list(ttnn.TensorAccessorArgs(pos_buf).get_compile_time_args())
    )
    prog = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=crs,
                compile_time_args=ct,
                defines=[("MODE_ATTN_SOFTMAX", "1")],
                runtime_args=rt,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=crs,
                compile_time_args=ct,
                defines=[("MODE_ATTN_SOFTMAX", "1"), ("IS_WRITER", "1")],
                runtime_args=rt,
                config=ttnn.WriterConfigDescriptor(),
            ),
        ],
        semaphores=[],
        cbs=cb_sm,
    )
    ttnn.generic_op([sc_raw, sink1, pos_buf], prog)
    return sc_raw
