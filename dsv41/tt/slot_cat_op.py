"""Fused 6-expert slot gather/concat + fused routed/shared SwiGLU wrapper for kernels/slot_cat.cpp."""
import os
import torch
import ttnn

_HERE = os.path.dirname(os.path.abspath(__file__))
KERNEL = os.path.join(os.path.dirname(_HERE), "kernels", "slot_cat.cpp")
PAGE16 = 2048


class SlotCatOp:
    def __init__(self, dev):
        self.dev = dev
        self.cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 4))])
        self.coords = [(x, y) for y in range(5) for x in range(8)]
        self.cores32 = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))])
        self.coords32 = [(x, y) for y in range(4) for x in range(8)]
        self.cbs = {}
        self._ct_cache = {}
        for fmt, page_w in (("bfp8", 1088), ("bfp4", 576)):
            dt = ttnn.bfloat8_b if fmt == "bfp8" else ttnn.bfloat4_b
            self.cbs[fmt] = [
                ttnn.CBDescriptor(
                    total_size=max(n_pages * page_w, PAGE16),
                    core_ranges=self.cores,
                    format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page_w)],
                )
                for i, n_pages in ((0, 108), (1, 162), (16, 54))
            ]
        self.cbs_swiglu = [
            ttnn.CBDescriptor(
                total_size=3 * PAGE16,
                core_ranges=self.cores32,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=PAGE16)],
            )
            for i in (0, 1)
        ]
        self.ctrl_dev = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, 1, 128]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, self.dev, ttnn.DRAM_MEMORY_CONFIG
        )
        self.h_sh_bufs = {
            s: ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, s, 288]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.dev, ttnn.DRAM_MEMORY_CONFIG
            )
            for s in range(1, 9)
        }
        self.h_sh_buf = self.h_sh_bufs[1]
        self._ctrl_bf16 = torch.zeros(1, 1, 1, 128, dtype=torch.bfloat16)
        self._ctrl_u16 = self._ctrl_bf16.view(torch.uint16)[0, 0, 0, :60].view(6, 10)
        self._ctrl_wt_view = self._ctrl_bf16[0, 0, 0, :60].view(6, 10)[:, 9]
        self._ctrl_wt_multi = self._ctrl_bf16[0, 0, 0, 64:112].view(6, 8)
        self._rep = ttnn.ReplicateTensorToMesh(dev)

    def _fill_slot_addrs(self, slots: list):
        for k, s in enumerate(slots):
            row9 = s.get("ctrl_row9")
            if row9 is None:
                a1, a3, a2 = s.get("addrs") or (
                    int(s["w1"].buffer_address()),
                    int(s["w3"].buffer_address()),
                    int(s["w2"].buffer_address()),
                )
                row9 = torch.tensor(
                    [
                        0x4000 | (a1 & 0x3FFF), 0x4000 | ((a1 >> 14) & 0x3FFF), 0x4000 | ((a1 >> 28) & 0xF),
                        0x4000 | (a3 & 0x3FFF), 0x4000 | ((a3 >> 14) & 0x3FFF), 0x4000 | ((a3 >> 28) & 0xF),
                        0x4000 | (a2 & 0x3FFF), 0x4000 | ((a2 >> 14) & 0x3FFF), 0x4000 | ((a2 >> 28) & 0xF),
                    ],
                    dtype=torch.uint16,
                )
                s["ctrl_row9"] = row9
            self._ctrl_u16[k, :9] = row9

    def update_ctrl(self, slots: list, wt_1d: torch.Tensor):
        """Write the 6 chosen slot buffer addresses and 6 routing weights into the fixed 256-byte ROW_MAJOR ctrl_dev buffer."""
        self._fill_slot_addrs(slots)
        self._ctrl_wt_view.copy_(wt_1d)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(self._ctrl_bf16, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self._rep),
            self.ctrl_dev,
        )

    def update_ctrl_multi(self, slots: list, wt_6_by_s: torch.Tensor):
        """Write 6 slot addresses and [6, s] multi-token routing weights (s <= 8) into ctrl_dev."""
        s = wt_6_by_s.shape[1]
        self._fill_slot_addrs(slots)
        self._ctrl_wt_view.copy_(wt_6_by_s[:, 0])
        self._ctrl_wt_multi[:, :s].copy_(wt_6_by_s)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(self._ctrl_bf16, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self._rep),
            self.ctrl_dev,
        )

    def run_from_ctrl(self, fmt: str, cat_dict: dict, wts_dev=None):
        """Trace-capturable 6-expert slot gather/concat reading slot addresses from self.ctrl_dev."""
        page_w = 1088 if fmt == "bfp8" else 576
        wts_addr = int(wts_dev.buffer_address()) if wts_dev is not None else 0
        base_args = [
            int(self.ctrl_dev.buffer_address()),
            int(cat_dict["w1"].buffer_address()),
            int(cat_dict["w3"].buffer_address()),
            int(cat_dict["w2"].buffer_address()),
            wts_addr,
        ]
        rt = ttnn.RuntimeArgs()
        for cid, (x, y) in enumerate(self.coords):
            rt[x][y] = base_args + [cid]

        ct = self._ct_cache.get(fmt)
        if ct is None:
            ct = (
                list(ttnn.TensorAccessorArgs(cat_dict["w1"]).get_compile_time_args())
                + list(ttnn.TensorAccessorArgs(self.ctrl_dev).get_compile_time_args())
            )
            self._ct_cache[fmt] = ct

        prog = ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(
                    kernel_source=KERNEL,
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=self.cores,
                    compile_time_args=ct,
                    defines=[("PAGE_W", str(page_w)), ("FROM_CTRL", "1")],
                    runtime_args=rt,
                    config=ttnn.ReaderConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=KERNEL,
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=self.cores,
                    compile_time_args=ct,
                    defines=[("PAGE_W", str(page_w)), ("FROM_CTRL", "1"), ("IS_WRITER", "1")],
                    runtime_args=rt,
                    config=ttnn.WriterConfigDescriptor(),
                ),
            ],
            semaphores=[],
            cbs=self.cbs[fmt],
        )
        tensors = [self.ctrl_dev, cat_dict["w1"], cat_dict["w3"], cat_dict["w2"]]
        if wts_dev is not None:
            tensors.append(wts_dev)
        ttnn.generic_op(tensors, prog)

    def swiglu_from_ctrl(self, g, u, gu_sh=None, s_rows: int = 1):
        """Fused s-row routed SwiGLU (in-place on g, weighted by ctrl_dev weights) + optional shared SwiGLU."""
        h_sh_out = self.h_sh_bufs[s_rows]
        gu_sh_ref = gu_sh if gu_sh is not None else h_sh_out
        base_args = [
            int(self.ctrl_dev.buffer_address()),
            int(g.buffer_address()),
            int(u.buffer_address()),
            int(gu_sh.buffer_address()) if gu_sh is not None else 0,
            int(h_sh_out.buffer_address()),
        ]
        rt = ttnn.RuntimeArgs()
        for cid, (x, y) in enumerate(self.coords32):
            rt[x][y] = base_args + [cid, int(s_rows)]

        ct = self._ct_cache.get("swiglu")
        if ct is None:
            ct = (
                list(ttnn.TensorAccessorArgs(self.ctrl_dev).get_compile_time_args())
                + list(ttnn.TensorAccessorArgs(g).get_compile_time_args())
                + list(ttnn.TensorAccessorArgs(u).get_compile_time_args())
                + list(ttnn.TensorAccessorArgs(gu_sh_ref).get_compile_time_args())
                + list(ttnn.TensorAccessorArgs(h_sh_out).get_compile_time_args())
            )
            self._ct_cache["swiglu"] = ct

        prog = ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(
                    kernel_source=KERNEL,
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=self.cores32,
                    compile_time_args=ct,
                    defines=[("MODE_SWIGLU", "1")],
                    runtime_args=rt,
                    config=ttnn.ReaderConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=KERNEL,
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=self.cores32,
                    compile_time_args=ct,
                    defines=[("MODE_SWIGLU", "1"), ("IS_WRITER", "1")],
                    runtime_args=rt,
                    config=ttnn.WriterConfigDescriptor(),
                ),
            ],
            semaphores=[],
            cbs=self.cbs_swiglu,
        )
        ttnn.generic_op([self.ctrl_dev, g, u, gu_sh_ref, h_sh_out], prog)
        return g, (h_sh_out if gu_sh is not None else None)

    def __call__(self, fmt: str, slots: list, cat_dict: dict, wts_dev, wt_1d: torch.Tensor):
        self.update_ctrl(slots, wt_1d)
        self.run_from_ctrl(fmt, cat_dict, wts_dev)
