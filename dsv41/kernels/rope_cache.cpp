// SPDX-License-Identifier: Apache-2.0
// Fast s=1 decode kernels for DeepSeek-V4.1-Flash on Wormhole B0:
//   1. MODE_ROPE (default): In-place RoPE rotation of the last 64 dims (last 2 tiles per head) of x [1, H, 1, D].
//   2. MODE_ROPE_COPY: Fused 64-worker (32 cores x 2 RISCs) reshape ([1,1,1,4096] <-> [1,8,1,512]) + RoPE / inv-RoPE.
//   3. MODE_CACHE_WRITE: In-place single-row update cache[0, 0, j, :] = row[0, 0, 0, :] across Wt cores, with optional
//      fused in-L1 RoPE on the last 2 tiles (cid >= wt - 2).
//   4. MODE_ATTN_SOFTMAX: Fused s=1 scale (512^-0.5) + causal/reach mask + sink logit + softmax in-place on [1, 8, 1, k]
//      across 8 cores (1 core/head) x 2 RISCs (RISCV_0 computes active tiles in Q20; RISCV_1 zeroes inactive tiles).
//   5. MODE_TILIZE_ROW0: 64-worker device tilizer from 1-page ROW_MAJOR [1,1,1,D] into TILE [1,HC,1,D] (+ optional pre0).
//   6. MODE_UNTILIZE_ROW0: 64-worker device untilizer of row 0 from TILE [1,1,1,D] into 1-page ROW_MAJOR [1,1,1,D].
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

static inline float bf16_to_f32(uint16_t b) {
    union { uint32_t u; float f; } v = { ((uint32_t)b) << 16 };
    return v.f;
}

static inline uint16_t f32_to_bf16(float f) {
    union { float f; uint32_t u; } v = { f };
    return (uint16_t)((v.u + 0x7FFFu + ((v.u >> 16) & 1u)) >> 16);
}

static inline void rotate_tile_rows(
    volatile tt_l1_ptr uint16_t* xp,
    volatile tt_l1_ptr uint16_t* cp,
    volatile tt_l1_ptr uint16_t* sp,
    float sign,
    uint32_t s_rows
) {
    for (uint32_t r = 0; r < s_rows; ++r) {
        const uint32_t r_off = (r < 16u) ? (r * 16u) : (512u + (r - 16u) * 16u);
        for (uint32_t face = 0; face < 2u; ++face) {
            const uint32_t base = r_off + face * 256u;
            for (uint32_t i = 0; i < 16u; i += 2u) {
                const uint32_t idx = base + i;
                const float x0 = bf16_to_f32(xp[idx]);
                const float x1 = bf16_to_f32(xp[idx + 1u]);
                const float c0 = bf16_to_f32(cp[idx]);
                const float c1 = bf16_to_f32(cp[idx + 1u]);
                const float s0 = bf16_to_f32(sp[idx]) * sign;
                const float s1 = bf16_to_f32(sp[idx + 1u]) * sign;
                xp[idx]      = f32_to_bf16(x0 * c0 - x1 * s0);
                xp[idx + 1u] = f32_to_bf16(x1 * c1 + x0 * s1);
            }
        }
    }
}

#if defined(MODE_ATTN_SOFTMAX)
static inline int32_t bf16_to_q20(uint16_t b) {
    uint32_t u = ((uint32_t)(b & 0x7FFFu)) << 16;
    if (u == 0) return 0;
    int32_t e = (int32_t)((u >> 23) & 0xFF) - 127;
    uint32_t m = (u & 0x7FFFFFu) | 0x800000u;
    int32_t q;
    if (e - 3 >= 7) {
        q = 0x3FFFFFFF;
    } else if (e - 3 >= 0) {
        q = (int32_t)(m << (e - 3));
    } else {
        int32_t sh = 3 - e;
        q = sh >= 24 ? 0 : (int32_t)(m >> sh);
    }
    return (b & 0x8000u) ? -q : q;
}

static inline uint16_t uq20_to_bf16(uint32_t q) {
    if (q == 0) return 0;
    uint32_t p = 31 - __builtin_clz(q);
    uint32_t man = p >= 23 ? (q >> (p - 23)) : (q << (23 - p));
    uint32_t u = ((p + 107) << 23) | (man & 0x7FFFFFu);
    return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

static inline uint16_t sq20_to_bf16(int32_t q) {
    if (q == 0) return 0;
    uint32_t sign = 0;
    uint32_t uq = (uint32_t)q;
    if (q < 0) {
        sign = 0x8000u;
        uq = (uint32_t)(-q);
    }
    return (uint16_t)(sign | uq20_to_bf16(uq));
}

static inline uint32_t exp_neg_q20(int32_t x_q20) {
    if (x_q20 >= 0) return 1048576u;
    if (x_q20 < -(20 << 20)) return 0u;
    int32_t t = (int32_t)(((int64_t)x_q20 * 1512775LL) >> 20);
    int32_t k = t >> 20;
    uint32_t f = (uint32_t)(t & 0xFFFFF);
    uint32_t p = 1398u;
    p = 10085u + (uint32_t)(((uint64_t)p * f) >> 20);
    p = 58201u + (uint32_t)(((uint64_t)p * f) >> 20);
    p = 251904u + (uint32_t)(((uint64_t)p * f) >> 20);
    p = 726817u + (uint32_t)(((uint64_t)p * f) >> 20);
    p = 1048576u + (uint32_t)(((uint64_t)p * f) >> 20);
    int32_t sh = -k;
    return sh >= 32 ? 0u : (p >> sh);
}

static inline int32_t scale_q20(uint16_t b) {
    // Multiply by 512^-0.5 = 2^-5 * sqrt(2) and round to bf16 to match ttnn.mul(sc, scale) without saturating at 1024
    uint32_t u = ((uint32_t)(b & 0x7FFFu)) << 16;
    if (u == 0) return 0;
    int32_t e = (int32_t)((u >> 23) & 0xFF) - 127 - 5;
    uint32_t m = (u & 0x7FFFFFu) | 0x800000u;
    uint32_t ms = (uint32_t)(((uint64_t)m * 1482910ULL) >> 20);   // m * sqrt(2)
    int32_t q;
    if (e - 3 >= 6) {
        q = 0x3FFFFFFF;
    } else if (e - 3 >= 0) {
        q = (int32_t)(ms << (e - 3));
    } else {
        int32_t sh = 3 - e;
        q = sh >= 25 ? 0 : (int32_t)(ms >> sh);
    }
    int32_t sq = (b & 0x8000u) ? -q : q;
    return bf16_to_q20(sq20_to_bf16(sq));
}
#endif

void kernel_main() {
    constexpr uint32_t PAGE = 2048;

#if defined(MODE_TILIZE_ROW0)
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t pre_addr = get_arg_val<uint32_t>(2);
    const uint32_t n_tiles  = get_arg_val<uint32_t>(3);
    const uint32_t hc       = get_arg_val<uint32_t>(4);
    const uint32_t cid      = get_arg_val<uint32_t>(5);   // 0..31

    constexpr auto src_args = TensorAccessorArgs<0>();
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();

    const auto src_ac = TensorAccessor(src_args, src_addr);
    const auto dst_ac = TensorAccessor(dst_args, dst_addr, PAGE);

#if !defined(IS_WRITER)
    constexpr uint32_t cb_id = 0;
    const uint32_t wid = cid;           // 0..31
#else
    constexpr uint32_t cb_id = 1;
    const uint32_t wid = 32u + cid;     // 32..63
#endif

    const uint32_t t_start = wid * 3u;
    if (t_start < n_tiles) {
        const uint32_t t_end = (t_start + 3u < n_tiles) ? (t_start + 3u) : n_tiles;
        const uint32_t cnt   = t_end - t_start;

        cb_reserve_back(cb_id, 5);
        const uint32_t buf_rm    = get_write_ptr(cb_id);
        const uint32_t buf_tiles = buf_rm + PAGE;

        const uint64_t src_noc_addr = src_ac.get_noc_addr(0) + (uint64_t)(t_start * 64u);
        noc_async_read(src_noc_addr, buf_rm, cnt * 64u);

        volatile tt_l1_ptr uint32_t* z32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf_tiles);
        for (uint32_t i = 0; i < 4u * 512u; ++i) z32[i] = 0;
        noc_async_read_barrier();

        volatile tt_l1_ptr uint32_t* rm32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf_rm);
        for (uint32_t k = 0; k < cnt; ++k) {
            const uint32_t t_page = buf_tiles + k * PAGE;
            volatile tt_l1_ptr uint32_t* tile32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(t_page);
            const uint32_t base32 = k * 16u;
            for (uint32_t i = 0; i < 8u; ++i) {
                tile32[i]        = rm32[base32 + i];
                tile32[128u + i] = rm32[base32 + 8u + i];
            }
            for (uint32_t c = 0; c < hc; ++c) {
                noc_async_write_page(c * n_tiles + t_start + k, dst_ac, t_page);
            }
        }
        if (pre_addr != 0 && wid < 4u) {
            const auto pre_ac = TensorAccessor(dst_args, pre_addr, PAGE);
            const uint32_t pre_page = buf_tiles + 3u * PAGE;
            volatile tt_l1_ptr uint16_t* tile16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(pre_page);
            tile16[0] = (wid == 0u) ? 0x3F80u : 0u;
            noc_async_write_page(wid, pre_ac, pre_page);
        }
        noc_async_write_barrier();
    }

#elif defined(MODE_UNTILIZE_ROW0)
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t nt_pack  = get_arg_val<uint32_t>(2);
    const uint32_t cid      = get_arg_val<uint32_t>(3);   // 0..31
    const uint32_t n_tiles  = nt_pack & 0xFFFFu;
    const uint32_t s_rows   = (nt_pack >> 16) ? (nt_pack >> 16) : 1u;

    constexpr auto src_args = TensorAccessorArgs<0>();
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();

    const auto src_ac = TensorAccessor(src_args, src_addr, PAGE);
    const auto dst_ac = TensorAccessor(dst_args, dst_addr);

#if !defined(IS_WRITER)
    constexpr uint32_t cb_id = 0;
    const uint32_t wid = cid;           // 0..31
#else
    constexpr uint32_t cb_id = 1;
    const uint32_t wid = 32u + cid;     // 32..63
#endif

    const uint32_t t_start = wid * 8u;
    if (t_start < n_tiles) {
        const uint32_t t_end = (t_start + 8u < n_tiles) ? (t_start + 8u) : n_tiles;
        const uint32_t cnt   = t_end - t_start;

        cb_reserve_back(cb_id, 9);
        const uint32_t buf_out   = get_write_ptr(cb_id);
        const uint32_t buf_tiles = buf_out + PAGE;

        for (uint32_t k = 0; k < cnt; ++k) {
            noc_async_read_page(t_start + k, src_ac, buf_tiles + k * PAGE);
        }
        noc_async_read_barrier();

        volatile tt_l1_ptr uint32_t* out32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf_out);
        for (uint32_t r = 0; r < s_rows; ++r) {
            const uint32_t f0 = (r < 16u) ? (r * 8u) : (256u + (r - 16u) * 8u);
            const uint32_t f1 = f0 + 128u;
            for (uint32_t k = 0; k < cnt; ++k) {
                volatile tt_l1_ptr uint32_t* t32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf_tiles + k * PAGE);
                const uint32_t base32 = k * 16u;
                for (uint32_t i = 0; i < 8u; ++i) {
                    out32[base32 + i]      = t32[f0 + i];
                    out32[base32 + 8u + i] = t32[f1 + i];
                }
            }
            const uint64_t dst_noc_addr = dst_ac.get_noc_addr(r) + (uint64_t)(t_start * 64u);
            noc_async_write(buf_out, dst_noc_addr, cnt * 64u);
            noc_async_write_barrier();
        }
    }

#elif defined(MODE_ATTN_SOFTMAX)
    const uint32_t sc_addr   = get_arg_val<uint32_t>(0);
    const uint32_t sink_addr = get_arg_val<uint32_t>(1);
    const uint32_t pos_addr  = get_arg_val<uint32_t>(2);
    const uint32_t kt        = get_arg_val<uint32_t>(3);
    const uint32_t mode_slot = get_arg_val<uint32_t>(4);
    const uint32_t h         = get_arg_val<uint32_t>(5);   // 0..7

    constexpr auto sc_args   = TensorAccessorArgs<0>();
    constexpr auto sink_args = TensorAccessorArgs<sc_args.next_compile_time_args_offset()>();
    constexpr auto pos_args  = TensorAccessorArgs<sink_args.next_compile_time_args_offset()>();

    const auto sc_ac  = TensorAccessor(sc_args, sc_addr, PAGE);
    const auto pos_ac = TensorAccessor(pos_args, pos_addr, PAGE);
    const uint32_t base_page = h * kt;

#if !defined(IS_WRITER)
    const auto sink_ac = TensorAccessor(sink_args, sink_addr, PAGE);
    cb_reserve_back(0, 38);
    const uint32_t buf_pos   = get_write_ptr(0);
    const uint32_t buf_sink  = buf_pos + PAGE;
    const uint32_t buf_tiles = buf_pos + 2 * PAGE;

    noc_async_read_page(0, pos_ac, buf_pos);
    noc_async_read_page(h, sink_ac, buf_sink);
    noc_async_read_barrier();

    volatile tt_l1_ptr uint16_t* pos16  = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_pos);
    volatile tt_l1_ptr uint16_t* sink16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_sink);
    const uint32_t n_win = (uint32_t)(pos16[3] & 0x3FFFu);
    const uint32_t n_ckv = (mode_slot == 0u) ? 0u : (uint32_t)(pos16[mode_slot] & 0x3FFFu);
    const uint32_t n_win_tiles = (n_win + 31u) >> 5;
    const uint32_t n_ckv_tiles = (n_ckv + 31u) >> 5;

    for (uint32_t t = 0; t < n_win_tiles; ++t) {
        noc_async_read_page(base_page + t, sc_ac, buf_tiles + t * PAGE);
    }
    for (uint32_t t = 0; t < n_ckv_tiles; ++t) {
        noc_async_read_page(base_page + 4u + t, sc_ac, buf_tiles + (4u + t) * PAGE);
    }
    noc_async_read_barrier();

    const int32_t sink_q = bf16_to_q20(sink16[0]);
    int32_t max_q = sink_q;

    for (uint32_t t = 0; t < n_win_tiles; ++t) {
        volatile tt_l1_ptr uint16_t* tp = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_tiles + t * PAGE);
        const uint32_t col_lim = (t + 1u == n_win_tiles && (n_win & 31u)) ? (n_win & 31u) : 32u;
        for (uint32_t c = 0; c < col_lim; ++c) {
            const uint32_t idx = (c < 16u) ? c : (256u + c - 16u);
            const int32_t vq = scale_q20(tp[idx]);
            if (vq > max_q) max_q = vq;
        }
    }
    for (uint32_t t = 0; t < n_ckv_tiles; ++t) {
        volatile tt_l1_ptr uint16_t* tp = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_tiles + (4u + t) * PAGE);
        const uint32_t col_lim = (t + 1u == n_ckv_tiles && (n_ckv & 31u)) ? (n_ckv & 31u) : 32u;
        for (uint32_t c = 0; c < col_lim; ++c) {
            const uint32_t idx = (c < 16u) ? c : (256u + c - 16u);
            const int32_t vq = scale_q20(tp[idx]);
            if (vq > max_q) max_q = vq;
        }
    }

    uint64_t sum_exp = exp_neg_q20(sink_q - max_q);
    for (uint32_t t = 0; t < n_win_tiles; ++t) {
        volatile tt_l1_ptr uint16_t* tp = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_tiles + t * PAGE);
        const uint32_t col_lim = (t + 1u == n_win_tiles && (n_win & 31u)) ? (n_win & 31u) : 32u;
        for (uint32_t c = 0; c < col_lim; ++c) {
            const uint32_t idx = (c < 16u) ? c : (256u + c - 16u);
            sum_exp += exp_neg_q20(scale_q20(tp[idx]) - max_q);
        }
    }
    for (uint32_t t = 0; t < n_ckv_tiles; ++t) {
        volatile tt_l1_ptr uint16_t* tp = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_tiles + (4u + t) * PAGE);
        const uint32_t col_lim = (t + 1u == n_ckv_tiles && (n_ckv & 31u)) ? (n_ckv & 31u) : 32u;
        for (uint32_t c = 0; c < col_lim; ++c) {
            const uint32_t idx = (c < 16u) ? c : (256u + c - 16u);
            sum_exp += exp_neg_q20(scale_q20(tp[idx]) - max_q);
        }
    }

    const uint64_t recip = (1ULL << 40) / (sum_exp ? sum_exp : 1ULL);
    for (uint32_t t = 0; t < n_win_tiles; ++t) {
        volatile tt_l1_ptr uint16_t* tp = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_tiles + t * PAGE);
        const uint32_t col_lim = (t + 1u == n_win_tiles && (n_win & 31u)) ? (n_win & 31u) : 32u;
        for (uint32_t c = 0; c < 32u; ++c) {
            const uint32_t idx = (c < 16u) ? c : (256u + c - 16u);
            if (c < col_lim) {
                const uint32_t e = exp_neg_q20(scale_q20(tp[idx]) - max_q);
                tp[idx] = uq20_to_bf16((uint32_t)(((uint64_t)e * recip) >> 20));
            } else {
                tp[idx] = 0;
            }
        }
        noc_async_write_page(base_page + t, sc_ac, buf_tiles + t * PAGE);
    }
    for (uint32_t t = 0; t < n_ckv_tiles; ++t) {
        volatile tt_l1_ptr uint16_t* tp = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_tiles + (4u + t) * PAGE);
        const uint32_t col_lim = (t + 1u == n_ckv_tiles && (n_ckv & 31u)) ? (n_ckv & 31u) : 32u;
        for (uint32_t c = 0; c < 32u; ++c) {
            const uint32_t idx = (c < 16u) ? c : (256u + c - 16u);
            if (c < col_lim) {
                const uint32_t e = exp_neg_q20(scale_q20(tp[idx]) - max_q);
                tp[idx] = uq20_to_bf16((uint32_t)(((uint64_t)e * recip) >> 20));
            } else {
                tp[idx] = 0;
            }
        }
        noc_async_write_page(base_page + 4u + t, sc_ac, buf_tiles + (4u + t) * PAGE);
    }
    noc_async_write_barrier();
#else
    (void)sink_addr;
    cb_reserve_back(1, 2);
    const uint32_t buf_pos  = get_write_ptr(1);
    const uint32_t buf_zero = buf_pos + PAGE;

    noc_async_read_page(0, pos_ac, buf_pos);
    volatile tt_l1_ptr uint32_t* z32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf_zero);
    for (uint32_t i = 0; i < 512u; ++i) z32[i] = 0;
    noc_async_read_barrier();

    volatile tt_l1_ptr uint16_t* pos16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_pos);
    const uint32_t n_win = (uint32_t)(pos16[3] & 0x3FFFu);
    const uint32_t n_ckv = (mode_slot == 0u) ? 0u : (uint32_t)(pos16[mode_slot] & 0x3FFFu);
    const uint32_t n_win_tiles = (n_win + 31u) >> 5;
    const uint32_t n_ckv_tiles = (n_ckv + 31u) >> 5;

    for (uint32_t t = n_win_tiles; t < 4u; ++t) {
        noc_async_write_page(base_page + t, sc_ac, buf_zero);
    }
    if (mode_slot != 0u) {
        for (uint32_t t = 4u + n_ckv_tiles; t < kt; ++t) {
            noc_async_write_page(base_page + t, sc_ac, buf_zero);
        }
    }
    noc_async_write_barrier();
#endif

#elif defined(MODE_ROPE_COPY)
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t cos_addr = get_arg_val<uint32_t>(1);
    const uint32_t sin_addr = get_arg_val<uint32_t>(2);
    const uint32_t dst_addr = get_arg_val<uint32_t>(3);
    const uint32_t inv_s    = get_arg_val<uint32_t>(4);
    const uint32_t cid      = get_arg_val<uint32_t>(5);   // 0..31
    const uint32_t inv      = inv_s & 1u;
    const uint32_t s_rows   = (inv_s >> 1) ? (inv_s >> 1) : 1u;

    constexpr auto src_args = TensorAccessorArgs<0>();
    constexpr auto cos_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();
    constexpr auto dst_args = TensorAccessorArgs<sin_args.next_compile_time_args_offset()>();

    const auto src_ac = TensorAccessor(src_args, src_addr, PAGE);
    const auto dst_ac = TensorAccessor(dst_args, dst_addr, PAGE);

#if !defined(IS_WRITER)
    constexpr uint32_t cb_id = 0;
    const uint32_t wid = cid;           // 0..31
#else
    constexpr uint32_t cb_id = 1;
    const uint32_t wid = 32u + cid;     // 32..63
#endif

    cb_reserve_back(cb_id, 6);
    const uint32_t buf_0 = get_write_ptr(cb_id);
    const uint32_t buf_1 = buf_0 + PAGE;

    const uint32_t p0 = wid * 2u;
    const uint32_t p1 = p0 + 1u;

    noc_async_read_page(p0, src_ac, buf_0);
    noc_async_read_page(p1, src_ac, buf_1);

    if ((wid & 7u) == 7u) {
        const auto cos_ac = TensorAccessor(cos_args, cos_addr, PAGE);
        const auto sin_ac = TensorAccessor(sin_args, sin_addr, PAGE);
        const uint32_t buf_c0 = buf_0 + 2 * PAGE;
        const uint32_t buf_c1 = buf_0 + 3 * PAGE;
        const uint32_t buf_s0 = buf_0 + 4 * PAGE;
        const uint32_t buf_s1 = buf_0 + 5 * PAGE;
        noc_async_read_page(0, cos_ac, buf_c0);
        noc_async_read_page(1, cos_ac, buf_c1);
        noc_async_read_page(0, sin_ac, buf_s0);
        noc_async_read_page(1, sin_ac, buf_s1);
        noc_async_read_barrier();

        const float sign = inv ? -1.0f : 1.0f;
        rotate_tile_rows(
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_0),
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_c0),
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_s0),
            sign,
            s_rows
        );
        rotate_tile_rows(
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_1),
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_c1),
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_s1),
            sign,
            s_rows
        );
    } else {
        (void)cos_addr; (void)sin_addr; (void)inv_s;
        noc_async_read_barrier();
    }

    noc_async_write_page(p0, dst_ac, buf_0);
    noc_async_write_page(p1, dst_ac, buf_1);
    noc_async_write_barrier();

#elif defined(MODE_CACHE_WRITE)
    const uint32_t cache_addr = get_arg_val<uint32_t>(0);
    const uint32_t row_addr   = get_arg_val<uint32_t>(1);
    const uint32_t pos_addr   = get_arg_val<uint32_t>(2);
    const uint32_t slot_idx   = get_arg_val<uint32_t>(3);
    const uint32_t max_rows   = get_arg_val<uint32_t>(4);
    const uint32_t wt         = get_arg_val<uint32_t>(5);
    const uint32_t cos_addr   = get_arg_val<uint32_t>(6);
    const uint32_t sin_addr   = get_arg_val<uint32_t>(7);
    const uint32_t cid        = get_arg_val<uint32_t>(8);

    constexpr auto cache_args = TensorAccessorArgs<0>();
    constexpr auto row_args   = TensorAccessorArgs<cache_args.next_compile_time_args_offset()>();
    constexpr auto pos_args   = TensorAccessorArgs<row_args.next_compile_time_args_offset()>();

    const auto cache_ac = TensorAccessor(cache_args, cache_addr, PAGE);
    const auto row_ac   = TensorAccessor(row_args, row_addr, PAGE);
    const auto pos_ac   = TensorAccessor(pos_args, pos_addr, PAGE);

    cb_reserve_back(0, 6);
    const uint32_t buf_pos   = get_write_ptr(0);
    const uint32_t buf_row   = buf_pos + PAGE;
    const uint32_t buf_cache = buf_pos + 2 * PAGE;

    if (pos_addr != 0) {
        noc_async_read_page(0, pos_ac, buf_pos);
    }
    noc_async_read_page(cid, row_ac, buf_row);
    if (cos_addr != 0 && cid >= wt - 2u) {
        const auto cos_ac = TensorAccessor(pos_args, cos_addr, PAGE);
        const auto sin_ac = TensorAccessor(pos_args, sin_addr, PAGE);
        const uint32_t rt = cid - (wt - 2u);
        noc_async_read_page(rt, cos_ac, buf_pos + 3 * PAGE);
        noc_async_read_page(rt, sin_ac, buf_pos + 4 * PAGE);
    }
    noc_async_read_barrier();

    uint32_t j0 = 0u;
    uint32_t s_rows = 1u;
    uint32_t is_ring = 0u;
    if (pos_addr != 0) {
        volatile tt_l1_ptr uint16_t* pos16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_pos);
        j0 = (uint32_t)(pos16[slot_idx] & 0x3FFFu);
    } else {
        j0 = slot_idx & 0xFFFFu;
        s_rows = (slot_idx >> 16) & 0xFFu;
        is_ring = (slot_idx >> 24) & 1u;
    }

    if (cos_addr != 0 && cid >= wt - 2u) {
        rotate_tile_rows(
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_row),
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_pos + 3 * PAGE),
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_pos + 4 * PAGE),
            1.0f,
            s_rows
        );
    }

    volatile tt_l1_ptr uint32_t* row32   = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf_row);
    volatile tt_l1_ptr uint32_t* cache32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf_cache);

    for (uint32_t r = 0; r < s_rows; ++r) {
        const uint32_t j = is_ring ? ((j0 + r) % max_rows) : (j0 + r);
        if (j < max_rows) {
            const uint32_t tr = j >> 5;
            const uint32_t r_in_tile = j & 31u;
            const uint32_t cache_page = tr * wt + cid;

            noc_async_read_page(cache_page, cache_ac, buf_cache);
            noc_async_read_barrier();

            const uint32_t f0_src = (r < 16u) ? (r * 8u) : (256u + (r - 16u) * 8u);
            const uint32_t f1_src = f0_src + 128u;
            const uint32_t f0_dst = (r_in_tile < 16u) ? (r_in_tile * 8u) : (256u + (r_in_tile - 16u) * 8u);
            const uint32_t f1_dst = f0_dst + 128u;
            for (uint32_t i = 0; i < 8u; ++i) {
                cache32[f0_dst + i] = row32[f0_src + i];
                cache32[f1_dst + i] = row32[f1_src + i];
            }

            noc_async_write_page(cache_page, cache_ac, buf_cache);
            noc_async_write_barrier();
        }
    }
#else
    // MODE_ROPE: In-place rotate last 2 tiles (64 dims) of head `cid` in x [1, H, s, D] (s <= 32)
    const uint32_t x_addr   = get_arg_val<uint32_t>(0);
    const uint32_t cos_addr = get_arg_val<uint32_t>(1);
    const uint32_t sin_addr = get_arg_val<uint32_t>(2);
    const uint32_t wt       = get_arg_val<uint32_t>(3);
    const uint32_t inv_s    = get_arg_val<uint32_t>(4);
    const uint32_t cid      = get_arg_val<uint32_t>(5);
    const uint32_t inv      = inv_s & 1u;
    const uint32_t s_rows   = (inv_s >> 1) ? (inv_s >> 1) : 1u;

    constexpr auto x_args   = TensorAccessorArgs<0>();
    constexpr auto cos_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();

    const auto x_ac   = TensorAccessor(x_args, x_addr, PAGE);
    const auto cos_ac = TensorAccessor(cos_args, cos_addr, PAGE);
    const auto sin_ac = TensorAccessor(sin_args, sin_addr, PAGE);

    cb_reserve_back(0, 6);
    const uint32_t buf_x0 = get_write_ptr(0);
    const uint32_t buf_x1 = buf_x0 + PAGE;
    const uint32_t buf_c0 = buf_x0 + 2 * PAGE;
    const uint32_t buf_c1 = buf_x0 + 3 * PAGE;
    const uint32_t buf_s0 = buf_x0 + 4 * PAGE;
    const uint32_t buf_s1 = buf_x0 + 5 * PAGE;

    const uint32_t p0 = cid * wt + (wt - 2u);
    const uint32_t p1 = p0 + 1u;

    noc_async_read_page(p0, x_ac, buf_x0);
    noc_async_read_page(p1, x_ac, buf_x1);
    noc_async_read_page(0, cos_ac, buf_c0);
    noc_async_read_page(1, cos_ac, buf_c1);
    noc_async_read_page(0, sin_ac, buf_s0);
    noc_async_read_page(1, sin_ac, buf_s1);
    noc_async_read_barrier();

    const float sign = inv ? -1.0f : 1.0f;
    for (uint32_t t = 0; t < 2u; ++t) {
        volatile tt_l1_ptr uint16_t* xp = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(t == 0 ? buf_x0 : buf_x1);
        volatile tt_l1_ptr uint16_t* cp = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(t == 0 ? buf_c0 : buf_c1);
        volatile tt_l1_ptr uint16_t* sp = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(t == 0 ? buf_s0 : buf_s1);
        rotate_tile_rows(xp, cp, sp, sign, s_rows);
    }

    noc_async_write_page(p0, x_ac, buf_x0);
    noc_async_write_page(p1, x_ac, buf_x1);
    noc_async_write_barrier();
#endif
}
