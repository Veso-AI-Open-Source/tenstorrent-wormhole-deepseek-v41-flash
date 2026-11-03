"""ttnn.generic_op wrapper for kernels/sinkhorn.cpp and kernels/mhc_fused.cpp."""
import os
import ttnn

_HERE = os.path.dirname(os.path.abspath(__file__))
KERNEL = os.path.join(os.path.dirname(_HERE), "kernels", "sinkhorn.cpp")
KERNEL_STREAM = os.path.join(os.path.dirname(_HERE), "kernels", "mhc_fused.cpp")
PAGE = 4096
PAGE16 = 2048


class SinkhornOp:
    def __init__(self, dev, iters=20):
        self.dev, self.iters = dev, iters
        self.cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
        self.cores40 = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 4))])
        self.coords40 = [(x, y) for y in range(5) for x in range(8)]
        self.cbs = [ttnn.CBDescriptor(total_size=PAGE, core_ranges=self.cores, format_descriptors=[
                        ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=PAGE)]) for i in (0, 16)]
        self.cbs16 = [
            ttnn.CBDescriptor(total_size=96 * PAGE16, core_ranges=self.cores, format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=PAGE16)]),
            ttnn.CBDescriptor(total_size=PAGE16, core_ranges=self.cores, format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=16, data_format=ttnn.bfloat16, page_size=PAGE16)]),
        ]
        self.cbs_stream = [
            ttnn.CBDescriptor(total_size=16 * PAGE16, core_ranges=self.cores40, format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=PAGE16)])
            for i in (0, 1)
        ]
        self.outs = {}                                          # n -> persistent [1,n,4,4] fp32 output (fixed address)
        self.out = self._out(1)
        self._slot = 0
        self.mix_outs = [
            (
                ttnn.allocate_tensor_on_device(ttnn.Shape([1, 4, 1, 1]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.dev, ttnn.DRAM_MEMORY_CONFIG),
                ttnn.allocate_tensor_on_device(ttnn.Shape([1, 4, 1, 1]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.dev, ttnn.DRAM_MEMORY_CONFIG),
                ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 4, 4]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.dev, ttnn.DRAM_MEMORY_CONFIG),
            )
            for _ in range(2)
        ]
        self.h_pre = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 1, 5120]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.dev, ttnn.DRAM_MEMORY_CONFIG)
        self.ms_buf = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 80, 1, 1]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.dev, ttnn.DRAM_MEMORY_CONFIG)
        self.x_mid = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 4, 1, 5120]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.dev, ttnn.DRAM_MEMORY_CONFIG)
        self.x_post = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 4, 1, 5120]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.dev, ttnn.DRAM_MEMORY_CONFIG)
        self._s_bufs = {}
        self._ct_cache = {}
        for s_pre in range(2, 7):
            self._get_s_bufs(s_pre)

    def _get_s_bufs(self, s: int):
        if s not in self._s_bufs:
            self._s_bufs[s] = {
                "mix": [
                    (
                        ttnn.allocate_tensor_on_device(ttnn.Shape([1, 4, s, 1]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.dev, ttnn.DRAM_MEMORY_CONFIG),
                        ttnn.allocate_tensor_on_device(ttnn.Shape([1, 4, s, 1]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.dev, ttnn.DRAM_MEMORY_CONFIG),
                        ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 4, 4]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.dev, ttnn.DRAM_MEMORY_CONFIG),
                    )
                    for _ in range(2)
                ],
                "h_pre": ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, s, 5120]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.dev, ttnn.DRAM_MEMORY_CONFIG),
                "x_mid": ttnn.allocate_tensor_on_device(ttnn.Shape([1, 4, s, 5120]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.dev, ttnn.DRAM_MEMORY_CONFIG),
                "x_post": ttnn.allocate_tensor_on_device(ttnn.Shape([1, 4, s, 5120]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.dev, ttnn.DRAM_MEMORY_CONFIG),
            }
        return self._s_bufs[s]

    def _out(self, n):
        if n not in self.outs:
            self.outs[n] = ttnn.allocate_tensor_on_device(ttnn.Shape([1, n, 4, 4]), ttnn.float32, ttnn.TILE_LAYOUT, self.dev, ttnn.DRAM_MEMORY_CONFIG)
        return self.outs[n]

    def mixes_1(self, mm4, ms, scale, base):
        """Fused s<=8 decode mHC sum(mm4) + rsqrt(ms+eps) + scale/base + sigmoids + softmax + 20-iter Sinkhorn + transpose:
        mm4 [1,4,s,24] bf16, ms [1,1..80,s,1] bf16, scale [1,1,1,24] bf16, base [1,1,1,24] bf16
        -> (pre [1,4,s,1] bf16, post [1,4,s,1] bf16, comb_t [1,1,4,4] bf16)."""
        s = int(mm4.shape[2])
        if s == 1:
            pre, post, comb_t = self.mix_outs[self._slot]
        else:
            pre, post, comb_t = self._get_s_bufs(s)["mix"][self._slot]
        self._slot ^= 1
        ms_pages = int(ms.shape[1])
        rt = ttnn.RuntimeArgs()
        rt[0][0] = [
            mm4.buffer_address(),
            ms.buffer_address(),
            scale.buffer_address(),
            base.buffer_address(),
            pre.buffer_address(),
            post.buffer_address(),
            comb_t.buffer_address(),
            self.iters,
            ms_pages,
            s,
        ]
        ct = self._ct_cache.get(("mixes_1", ms_pages))
        if ct is None:
            ct = (
                list(ttnn.TensorAccessorArgs(mm4).get_compile_time_args())
                + list(ttnn.TensorAccessorArgs(ms).get_compile_time_args())
                + list(ttnn.TensorAccessorArgs(scale).get_compile_time_args())
                + list(ttnn.TensorAccessorArgs(base).get_compile_time_args())
                + list(ttnn.TensorAccessorArgs(pre).get_compile_time_args())
                + list(ttnn.TensorAccessorArgs(post).get_compile_time_args())
                + list(ttnn.TensorAccessorArgs(comb_t).get_compile_time_args())
            )
            self._ct_cache[("mixes_1", ms_pages)] = ct
        prog = ttnn.ProgramDescriptor(
            kernels=[ttnn.KernelDescriptor(
                kernel_source=KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=self.cores,
                compile_time_args=ct,
                defines=[("MHC_FUSED", "1")],
                runtime_args=rt,
                config=ttnn.ReaderConfigDescriptor(),
            )],
            semaphores=[],
            cbs=self.cbs16,
        )
        ttnn.generic_op([mm4, ms, scale, base, pre, post, comb_t], prog)
        return pre, post, comb_t

    def _run_stream(self, mode_define: str, ct_key: str, tensors: list, s_rows: int = 1):
        rt = ttnn.RuntimeArgs()
        base_args = [int(t.buffer_address()) for t in tensors]
        for cid, (cx, cy) in enumerate(self.coords40):
            rt[cx][cy] = base_args + [cid, int(s_rows)]
        ct = self._ct_cache.get(ct_key)
        if ct is None:
            ct = []
            for t in tensors:
                ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
            self._ct_cache[ct_key] = ct
        prog = ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(
                    kernel_source=KERNEL_STREAM,
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=self.cores40,
                    compile_time_args=ct,
                    defines=[(mode_define, "1")],
                    runtime_args=rt,
                    config=ttnn.ReaderConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=KERNEL_STREAM,
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=self.cores40,
                    compile_time_args=ct,
                    defines=[(mode_define, "1"), ("IS_WRITER", "1")],
                    runtime_args=rt,
                    config=ttnn.WriterConfigDescriptor(),
                ),
            ],
            semaphores=[],
            cbs=self.cbs_stream,
        )
        ttnn.generic_op(tensors, prog)

    def pre_ms(self, x, pre):
        """Fused s<=8 MHC.pre(x, pre) + partial sum(x^2) across 40 cores x 2 RISCs (2 col tiles/worker).
        Returns (h_pre [1,1,s,5120], ms_buf [1,80,1,1])."""
        s = int(x.shape[2])
        h_pre = self.h_pre if s == 1 else self._get_s_bufs(s)["h_pre"]
        self._run_stream("MODE_PRE_MS", "pre_ms", [x, pre, h_pre, self.ms_buf], s_rows=s)
        return h_pre, self.ms_buf

    def post_pre_ms(self, y, res, post, comb_t, pre, x_out=None):
        """Fused s<=8 MHC.post(y, res, post, comb_t) + MHC.pre(x_out, pre) + partial sum(x_out^2) across 80 workers.
        Returns (x_out [1,4,s,5120], h_pre [1,1,s,5120], ms_buf [1,80,1,1])."""
        s = int(res.shape[2])
        if x_out is None:
            x_out = self.x_mid if s == 1 else self._get_s_bufs(s)["x_mid"]
        h_pre = self.h_pre if s == 1 else self._get_s_bufs(s)["h_pre"]
        self._run_stream("MODE_POST_PRE_MS", "post_pre_ms", [y, res, post, comb_t, pre, x_out, h_pre, self.ms_buf], s_rows=s)
        return x_out, h_pre, self.ms_buf

    def post_only(self, y, res, post, comb_t, x_out=None):
        """Fused s<=8 MHC.post(y, res, post, comb_t) across 40 cores x 2 RISCs (2 col tiles/worker).
        Returns x_out [1,4,s,5120]."""
        s = int(res.shape[2])
        if x_out is None:
            x_out = self.x_post if s == 1 else self._get_s_bufs(s)["x_post"]
        self._run_stream("MODE_POST_ONLY", "post_only", [y, res, post, comb_t, x_out], s_rows=s)
        return x_out

    def __call__(self, comb_bf16):
        """comb [1,n,4,4] bf16 (= softmax + eps), one tile per matrix -> doubly stochastic [1,n,4,4] bf16.
        One launch for all n (a serial fixed-point loop on one RISC: 39 x 16 divides per matrix)."""
        n = comb_bf16.shape[1]; out = self._out(n)
        x = ttnn.typecast(comb_bf16, ttnn.float32)
        rt = ttnn.RuntimeArgs(); rt[0][0] = [x.buffer_address(), out.buffer_address(), self.iters, n]
        ct = list(ttnn.TensorAccessorArgs(x).get_compile_time_args()) + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
        prog = ttnn.ProgramDescriptor(kernels=[ttnn.KernelDescriptor(
            kernel_source=KERNEL, source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=self.cores,
            compile_time_args=ct, runtime_args=rt, config=ttnn.ReaderConfigDescriptor())], semaphores=[], cbs=self.cbs)
        ttnn.generic_op([x, out], prog)
        return ttnn.typecast(out, ttnn.bfloat16)
