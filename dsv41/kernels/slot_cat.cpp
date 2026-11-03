// SPDX-License-Identifier: Apache-2.0
// Fused 6-expert slot gather/concat + fused routed/shared SwiGLU on Wormhole B0 (40 cores x 2 RISCs).
//   1. Default / FROM_CTRL: 50/50 balanced 6-expert gather across 40 cores (8x5) x 2 RISCs:
//      - RISCV_0 (Reader, NOC0): w1 (4 rows = 36 tiles/core/exp) + w3 rows 0..79 (2 rows = 18 tiles/core/exp) = 54 tiles/exp
//      - RISCV_1 (Writer, NOC1): w2 (36 tiles/core/exp) + w3 rows 80..159 (2 rows = 18 tiles/core/exp) = 54 tiles/exp
//   2. MODE_SWIGLU: Fused routed SwiGLU (54 tiles, weighted by wt_bf16[0..5] from ctrl_dev) + shared SwiGLU (9 tiles)
//      across 32 cores (8x4) x 2 RISCs = 64 workers (1 tile = 32 row-0 elements per worker) in pure RV32IM Q20 integer math.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

#ifndef PAGE_W
#define PAGE_W 1088
#endif

static inline uint32_t dec_addr(volatile tt_l1_ptr uint16_t* p) {
    return (uint32_t)(p[0] & 0x3FFFu) | ((uint32_t)(p[1] & 0x3FFFu) << 14) | ((uint32_t)(p[2] & 0xFu) << 28);
}

#if defined(MODE_SWIGLU)
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

static inline uint32_t sigmoid_q20(int32_t x_q20) {
    if (x_q20 >= 0) {
        uint32_t e = exp_neg_q20(-x_q20);
        return (uint32_t)((1ULL << 40) / (1048576u + e));
    } else {
        uint32_t e = exp_neg_q20(x_q20);
        return (uint32_t)(((uint64_t)e << 20) / (1048576u + e));
    }
}

void kernel_main() {
    constexpr uint32_t PAGE16 = 2048, PAGE_CTRL = 256;
    constexpr int32_t LIMIT_Q20 = 10 << 20;

    const uint32_t ctrl_addr  = get_arg_val<uint32_t>(0);
    const uint32_t g_addr     = get_arg_val<uint32_t>(1);
    const uint32_t u_addr     = get_arg_val<uint32_t>(2);
    const uint32_t gu_sh_addr = get_arg_val<uint32_t>(3);
    const uint32_t h_sh_addr  = get_arg_val<uint32_t>(4);
    const uint32_t cid        = get_arg_val<uint32_t>(5);   // 0..31
    const uint32_t s_rows     = get_arg_val<uint32_t>(6);   // 1..8

    constexpr auto ctrl_args  = TensorAccessorArgs<0>();
    constexpr auto g_args     = TensorAccessorArgs<ctrl_args.next_compile_time_args_offset()>();
    constexpr auto u_args     = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    constexpr auto gu_sh_args = TensorAccessorArgs<u_args.next_compile_time_args_offset()>();
    constexpr auto h_sh_args  = TensorAccessorArgs<gu_sh_args.next_compile_time_args_offset()>();

#if !defined(IS_WRITER)
    constexpr uint32_t cb_id = 0;
    const uint32_t wid = cid;           // 0..31
#else
    constexpr uint32_t cb_id = 1;
    const uint32_t wid = 32u + cid;     // 32..63
#endif

    cb_reserve_back(cb_id, 3);
    const uint32_t buf_g    = get_write_ptr(cb_id);
    const uint32_t buf_u    = buf_g + PAGE16;
    const uint32_t buf_ctrl = buf_g + 2 * PAGE16;

    if (wid < 54u) {
        const auto ctrl_ac = TensorAccessor(ctrl_args, ctrl_addr, PAGE_CTRL);
        const auto g_ac    = TensorAccessor(g_args, g_addr, PAGE16);
        const auto u_ac    = TensorAccessor(u_args, u_addr, PAGE16);

        noc_async_read_page(0, ctrl_ac, buf_ctrl);
        noc_async_read_page(wid, g_ac, buf_g);
        noc_async_read_page(wid, u_ac, buf_u);
        noc_async_read_barrier();

        volatile tt_l1_ptr uint16_t* ctrl16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_ctrl);
        volatile tt_l1_ptr uint16_t* gp     = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_g);
        volatile tt_l1_ptr uint16_t* up     = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_u);

        const uint32_t k = (wid * 57u) >> 9;   // exact wid / 9 for wid in 0..53

        for (uint32_t r = 0; r < s_rows; ++r) {
            const uint16_t wt_b = (r == 0u) ? ctrl16[k * 10u + 9u] : ctrl16[64u + k * 8u + r];
            const int32_t wt_q20 = bf16_to_q20(wt_b);
            if (wt_q20 == 0) {
                for (uint32_t face = 0; face < 2u; ++face) {
                    const uint32_t base = face * 256u + r * 16u;
                    for (uint32_t i = 0; i < 16u; ++i) gp[base + i] = 0;
                }
                continue;
            }
            for (uint32_t face = 0; face < 2u; ++face) {
                const uint32_t base = face * 256u + r * 16u;
                for (uint32_t i = 0; i < 16u; ++i) {
                    const uint32_t idx = base + i;
                    int32_t g_q = bf16_to_q20(gp[idx]);
                    if (g_q > LIMIT_Q20) g_q = LIMIT_Q20;
                    int32_t u_q = bf16_to_q20(up[idx]);
                    if (u_q > LIMIT_Q20) u_q = LIMIT_Q20;
                    if (u_q < -LIMIT_Q20) u_q = -LIMIT_Q20;

                    const uint32_t sig_q = sigmoid_q20(g_q);
                    const int32_t silu_q = bf16_to_q20(sq20_to_bf16((int32_t)(((int64_t)g_q * (int64_t)sig_q) >> 20)));
                    const int32_t prod_q = bf16_to_q20(sq20_to_bf16((int32_t)(((int64_t)silu_q * (int64_t)u_q) >> 20)));
                    const int32_t out_q  = (int32_t)(((int64_t)prod_q * (int64_t)wt_q20) >> 20);
                    gp[idx] = sq20_to_bf16(out_q);
                }
            }
        }
        noc_async_write_page(wid, g_ac, buf_g);
        noc_async_write_barrier();
    } else if (wid < 63u && gu_sh_addr != 0u) {
        const uint32_t sh_t = wid - 54u;
        const auto gu_sh_ac = TensorAccessor(gu_sh_args, gu_sh_addr, PAGE16);
        const auto h_sh_ac  = TensorAccessor(h_sh_args, h_sh_addr, PAGE16);

        noc_async_read_page(sh_t, gu_sh_ac, buf_g);
        noc_async_read_page(sh_t + 9u, gu_sh_ac, buf_u);
        noc_async_read_barrier();

        volatile tt_l1_ptr uint16_t* gp = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_g);
        volatile tt_l1_ptr uint16_t* up = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_u);

        for (uint32_t r = 0; r < s_rows; ++r) {
            for (uint32_t face = 0; face < 2u; ++face) {
                const uint32_t base = face * 256u + r * 16u;
                for (uint32_t i = 0; i < 16u; ++i) {
                    const uint32_t idx = base + i;
                    int32_t g_q = bf16_to_q20(gp[idx]);
                    if (g_q > LIMIT_Q20) g_q = LIMIT_Q20;
                    int32_t u_q = bf16_to_q20(up[idx]);
                    if (u_q > LIMIT_Q20) u_q = LIMIT_Q20;
                    if (u_q < -LIMIT_Q20) u_q = -LIMIT_Q20;

                    const uint32_t sig_q = sigmoid_q20(g_q);
                    const int32_t silu_q = bf16_to_q20(sq20_to_bf16((int32_t)(((int64_t)g_q * (int64_t)sig_q) >> 20)));
                    const int32_t prod_q = (int32_t)(((int64_t)silu_q * (int64_t)u_q) >> 20);
                    gp[idx] = sq20_to_bf16(prod_q);
                }
            }
        }
        noc_async_write_page(sh_t, h_sh_ac, buf_g);
        noc_async_write_barrier();
    }
}
#else
void kernel_main() {
    constexpr uint32_t PAGE16 = 2048, PAGE_CTRL = 256;
    constexpr auto w_args    = TensorAccessorArgs<0>();
    constexpr auto ctrl_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();

    uint32_t w1_addrs[6], w3_addrs[6], w2_addrs[6], wt_bf16[6];
#if defined(FROM_CTRL)
    const uint32_t ctrl_addr   = get_arg_val<uint32_t>(0);
    const uint32_t cat_w1_addr = get_arg_val<uint32_t>(1);
    const uint32_t cat_w3_addr = get_arg_val<uint32_t>(2);
    const uint32_t cat_w2_addr = get_arg_val<uint32_t>(3);
    const uint32_t wts_addr    = get_arg_val<uint32_t>(4);
    const uint32_t cid         = get_arg_val<uint32_t>(5);   // 0..39
    const auto ctrl_ac = TensorAccessor(ctrl_args, ctrl_addr, PAGE_CTRL);
#else
    for (uint32_t k = 0; k < 6; ++k) w1_addrs[k] = get_arg_val<uint32_t>(k);
    for (uint32_t k = 0; k < 6; ++k) w3_addrs[k] = get_arg_val<uint32_t>(6 + k);
    for (uint32_t k = 0; k < 6; ++k) w2_addrs[k] = get_arg_val<uint32_t>(12 + k);
    const uint32_t cat_w1_addr = get_arg_val<uint32_t>(18);
    const uint32_t cat_w3_addr = get_arg_val<uint32_t>(19);
    const uint32_t cat_w2_addr = get_arg_val<uint32_t>(20);
    const uint32_t wts_addr    = get_arg_val<uint32_t>(21);
    for (uint32_t k = 0; k < 6; ++k) wt_bf16[k] = get_arg_val<uint32_t>(22 + k);
    const uint32_t cid         = get_arg_val<uint32_t>(28);   // 0..39
#endif

#if !defined(IS_WRITER)
    // RISCV_0 (NOC_0): w1 (4 tile rows = 36 tiles) + w3 rows 0..79 (2 tile rows = 18 tiles), batched 3 experts/barrier
    constexpr uint32_t cb_a = 0, cb_b = 16;
    const auto cat_w1 = TensorAccessor(w_args, cat_w1_addr, PAGE_W);
    const auto cat_w3 = TensorAccessor(w_args, cat_w3_addr, PAGE_W);

    cb_reserve_back(cb_a, 108);
    cb_reserve_back(cb_b, 54);
    const uint32_t buf_a = get_write_ptr(cb_a);
    const uint32_t buf_b = get_write_ptr(cb_b);

#if defined(FROM_CTRL)
    noc_async_read_page(0, ctrl_ac, buf_a);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint16_t* ctrl16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_a);
    for (uint32_t k = 0; k < 6; ++k) {
        volatile tt_l1_ptr uint16_t* row = ctrl16 + k * 10u;
        w1_addrs[k] = dec_addr(row + 0);
        w3_addrs[k] = dec_addr(row + 3);
    }
#endif

    const uint32_t r1_start = cid * 4;
    const uint32_t r3_start = cid * 2;
    for (uint32_t eb = 0; eb < 6; eb += 3) {
        for (uint32_t dk = 0; dk < 3; ++dk) {
            const uint32_t k = eb + dk;
            const auto in_w1 = TensorAccessor(w_args, w1_addrs[k], PAGE_W);
            const auto in_w3 = TensorAccessor(w_args, w3_addrs[k], PAGE_W);
            const uint32_t a_exp = buf_a + dk * 36 * PAGE_W;
            const uint32_t b_exp = buf_b + dk * 18 * PAGE_W;
            for (uint32_t dr = 0; dr < 4; ++dr) {
                const uint32_t src_base = (r1_start + dr) * 9;
                const uint32_t l1_off = a_exp + (dr * 9) * PAGE_W;
                for (uint32_t c = 0; c < 9; ++c) {
                    noc_async_read_page(src_base + c, in_w1, l1_off + c * PAGE_W);
                }
            }
            for (uint32_t dr = 0; dr < 2; ++dr) {
                const uint32_t src_base = (r3_start + dr) * 9;
                const uint32_t l1_off = b_exp + (dr * 9) * PAGE_W;
                for (uint32_t c = 0; c < 9; ++c) {
                    noc_async_read_page(src_base + c, in_w3, l1_off + c * PAGE_W);
                }
            }
        }
        noc_async_read_barrier();
        for (uint32_t dk = 0; dk < 3; ++dk) {
            const uint32_t k = eb + dk;
            const uint32_t a_exp = buf_a + dk * 36 * PAGE_W;
            const uint32_t b_exp = buf_b + dk * 18 * PAGE_W;
            for (uint32_t dr = 0; dr < 4; ++dr) {
                const uint32_t dst_base = (r1_start + dr) * 54 + k * 9;
                const uint32_t l1_off = a_exp + (dr * 9) * PAGE_W;
                for (uint32_t c = 0; c < 9; ++c) {
                    noc_async_write_page(dst_base + c, cat_w1, l1_off + c * PAGE_W);
                }
            }
            for (uint32_t dr = 0; dr < 2; ++dr) {
                const uint32_t dst_base = (r3_start + dr) * 54 + k * 9;
                const uint32_t l1_off = b_exp + (dr * 9) * PAGE_W;
                for (uint32_t c = 0; c < 9; ++c) {
                    noc_async_write_page(dst_base + c, cat_w3, l1_off + c * PAGE_W);
                }
            }
        }
        noc_async_write_barrier();
    }
    (void)w2_addrs; (void)cat_w2_addr; (void)wts_addr; (void)wt_bf16;
#else
    // RISCV_1 (NOC_1): w2 (36 tiles/core) + w3 rows 80..159 (2 tile rows = 18 tiles/core), batched 3 experts/barrier
    constexpr uint32_t cb_c = 1;
    const auto cat_w2 = TensorAccessor(w_args, cat_w2_addr, PAGE_W);
    const auto cat_w3 = TensorAccessor(w_args, cat_w3_addr, PAGE_W);

    cb_reserve_back(cb_c, 162);
    const uint32_t buf_w2 = get_write_ptr(cb_c);
    const uint32_t buf_w3 = buf_w2 + 108 * PAGE_W;

#if defined(FROM_CTRL)
    noc_async_read_page(0, ctrl_ac, buf_w2);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint16_t* ctrl16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_w2);
    for (uint32_t k = 0; k < 6; ++k) {
        volatile tt_l1_ptr uint16_t* row = ctrl16 + k * 10u;
        w3_addrs[k] = dec_addr(row + 3);
        w2_addrs[k] = dec_addr(row + 6);
        wt_bf16[k]  = row[9];
    }
#endif

    const uint32_t p_start  = cid * 36;
    const uint32_t r3_start = 80 + cid * 2;
    for (uint32_t eb = 0; eb < 6; eb += 3) {
        for (uint32_t dk = 0; dk < 3; ++dk) {
            const uint32_t k = eb + dk;
            const auto in_w2 = TensorAccessor(w_args, w2_addrs[k], PAGE_W);
            const auto in_w3 = TensorAccessor(w_args, w3_addrs[k], PAGE_W);
            const uint32_t w2_exp = buf_w2 + dk * 36 * PAGE_W;
            const uint32_t w3_exp = buf_w3 + dk * 18 * PAGE_W;
            for (uint32_t c = 0; c < 36; ++c) {
                noc_async_read_page(p_start + c, in_w2, w2_exp + c * PAGE_W);
            }
            for (uint32_t dr = 0; dr < 2; ++dr) {
                const uint32_t src_base = (r3_start + dr) * 9;
                const uint32_t l1_off = w3_exp + (dr * 9) * PAGE_W;
                for (uint32_t c = 0; c < 9; ++c) {
                    noc_async_read_page(src_base + c, in_w3, l1_off + c * PAGE_W);
                }
            }
        }
        noc_async_read_barrier();
        for (uint32_t dk = 0; dk < 3; ++dk) {
            const uint32_t k = eb + dk;
            const uint32_t dst_k = k * 1440 + p_start;
            const uint32_t w2_exp = buf_w2 + dk * 36 * PAGE_W;
            const uint32_t w3_exp = buf_w3 + dk * 18 * PAGE_W;
            for (uint32_t c = 0; c < 36; ++c) {
                noc_async_write_page(dst_k + c, cat_w2, w2_exp + c * PAGE_W);
            }
            for (uint32_t dr = 0; dr < 2; ++dr) {
                const uint32_t dst_base = (r3_start + dr) * 54 + k * 9;
                const uint32_t l1_off = w3_exp + (dr * 9) * PAGE_W;
                for (uint32_t c = 0; c < 9; ++c) {
                    noc_async_write_page(dst_base + c, cat_w3, l1_off + c * PAGE_W);
                }
            }
        }
        noc_async_write_barrier();
    }

    if (wts_addr != 0 && cid < 6) {
        const auto wts_ac = TensorAccessor(w_args, wts_addr, PAGE16);
        const uint16_t wb = (uint16_t)(wt_bf16[cid] & 0xFFFFu);
        const uint32_t w32 = ((uint32_t)wb) | (((uint32_t)wb) << 16);
        volatile tt_l1_ptr uint32_t* p32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf_w2);
        for (uint32_t i = 0; i < 512; ++i) p32[i] = w32;
        const uint32_t base_tile = cid * 9;
        for (uint32_t c = 0; c < 9; ++c) {
            noc_async_write_page(base_tile + c, wts_ac, buf_w2);
        }
        noc_async_write_barrier();
    }
    (void)w1_addrs; (void)cat_w1_addr;
#endif
}
#endif
