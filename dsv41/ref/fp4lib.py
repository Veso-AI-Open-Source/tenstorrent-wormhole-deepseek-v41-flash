"""ctypes binding for libfp4.so (see fp4lib.c)."""
import ctypes, os
import torch

_lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libfp4.so"))
_lib.dequant_fp4.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long, ctypes.c_long]
_lib.bfp_roundtrip.argtypes = [ctypes.c_void_p, ctypes.c_long, ctypes.c_long, ctypes.c_int, ctypes.c_int]
_lib.pack_bfp_tiles_t.argtypes = [ctypes.c_void_p, ctypes.c_long, ctypes.c_long, ctypes.c_long, ctypes.c_long, ctypes.c_long, ctypes.c_int, ctypes.c_void_p]
_lib.bfp_roundtrip_t.argtypes = [ctypes.c_void_p, ctypes.c_long, ctypes.c_long, ctypes.c_int, ctypes.c_int]
if hasattr(_lib, "pack_expert_fp4_shards"):
    _lib.pack_expert_fp4_shards.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]


def dequant_fp4(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """q [rows, cols/2] (any 1-byte dtype), s [rows, cols/32] E8M0 bytes -> float32 [rows, cols]."""
    rows, half = q.shape
    assert q.is_contiguous() and s.is_contiguous() and s.shape == (rows, half * 2 // 32)
    w = torch.empty(rows, half * 2, dtype=torch.float32)
    _lib.dequant_fp4(q.data_ptr(), s.data_ptr(), w.data_ptr(), rows, half * 2)
    return w


def bfp_roundtrip_(w: torch.Tensor, mant_bits: int, mode: int) -> torch.Tensor:
    assert w.dtype == torch.float32 and w.is_contiguous() and w.dim() == 2 and w.size(1) % 16 == 0
    _lib.bfp_roundtrip(w.data_ptr(), w.size(0), w.size(1), mant_bits, mode)
    return w


def bfp_roundtrip_t_(w: torch.Tensor, mant_bits: int, mode: int) -> torch.Tensor:
    """In place, blocks of 16 running down the rows: equals bfp_roundtrip_ applied to w.T."""
    assert w.dtype == torch.float32 and w.is_contiguous() and w.dim() == 2 and w.size(0) % 16 == 0
    _lib.bfp_roundtrip_t(w.data_ptr(), w.size(0), w.size(1), mant_bits, mode)
    return w


def pack_tiles_t(W: torch.Tensor, i0: int, j0: int, rows: int, cols: int, mant_bits: int) -> torch.Tensor:
    """Packed ttnn tiles (uint8) of the device matrix D = W.T[i0:i0+rows, j0:j0+cols]. mant_bits 7=bfp8, 3=bfp4."""
    assert W.dtype == torch.float32 and W.is_contiguous() and rows % 32 == 0 and cols % 32 == 0
    out = torch.empty((rows // 32) * (cols // 32) * (64 + (1024 if mant_bits == 7 else 512)), dtype=torch.uint8)
    _lib.pack_bfp_tiles_t(W.data_ptr(), W.size(1), i0, j0, rows, cols, mant_bits, out.data_ptr())
    return out


def pack_expert_shards_into(q_u8, s_u8, is_w2: bool, mant_bits: int, out_u8, out_offset: int = 0) -> None:
    """Fused zero-copy L1-tiled dequant + 8-shard BFP8/BFP4 tile packer directly from mmap uint8 views."""
    _lib.pack_expert_fp4_shards(
        ctypes.c_void_p(q_u8.ctypes.data),
        ctypes.c_void_p(s_u8.ctypes.data),
        1 if is_w2 else 0,
        int(mant_bits),
        ctypes.c_void_p(out_u8.ctypes.data + out_offset),
    )
