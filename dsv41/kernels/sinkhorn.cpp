// SPDX-License-Identifier: Apache-2.0
// mHC Sinkhorn on ONE data-movement RISC: a 4x4 positive matrix -> doubly stochastic, exactly the
// reference sequence (kernel.py hc_split_sinkhorn): comb /= (colsum + eps); then (iters-1) x
// { comb /= (rowsum + eps); comb /= (colsum + eps) }.
// Why a kernel: as stock ops this is 79 launches of ~36 us each on a 16-element tensor (2.9 ms), twice
// per layer, 80 times per token. Here it is one launch and ~1e5 integer cycles.
// Arithmetic is Q20 fixed point (no FPU on these cores): one 64-bit reciprocal per row/column, then
// 32x32->64 multiplies. Resolution 1e-6, far below the bf16 the result is cast back to.
// io: fp32 TILE tensors [1,1,4,4] -> element (r,c) at float index r*16 + c of face 0 of tile 0.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

static inline uint32_t to_q20(uint32_t u) {            // fp32 bits (value in [0, 4)) -> Q20
    uint32_t e = (u >> 23) & 0xFF;
    if (e == 0) return 0;
    uint32_t m = (u & 0x7FFFFFu) | 0x800000u;          // value = m * 2^(e-150); Q20 = m * 2^(e-130)
    if (e >= 130) return m << (e - 130);
    uint32_t sh = 130 - e;
    return sh >= 32 ? 0 : (m >> sh);
}/**/
static inline uint32_t from_q20(uint32_t q) {          // Q20 -> fp32 bits
    if (q == 0) return 0;
    uint32_t p = 31 - __builtin_clz(q);
    uint32_t man = p >= 23 ? (q >> (p - 23)) : (q << (23 - p));
    return ((p + 107) << 23) | (man & 0x7FFFFFu);      // exponent = p - 20 + 127
}
static inline uint32_t qdiv(uint32_t v, uint64_t recip) {   // v * recip >> 20, recip = 2^40 / sum
    return (uint32_t)(((uint64_t)v * recip) >> 20);
}

#if defined(MHC_FUSED)
static inline float bf16_to_f32(uint16_t b) {
    union { uint32_t u; float f; } v = { ((uint32_t)b) << 16 };
    return v.f;
}

static inline uint16_t f32_to_bf16(float f) {
    union { float f; uint32_t u; } v = { f };
    return (uint16_t)((v.u + 0x7FFFu + ((v.u >> 16) & 1u)) >> 16);
}

static inline float fast_rsqrt(float x) {
    x += 1e-20f;
    union { float f; uint32_t u; } v = { x };
    v.u = 0x5f3759dfu - (v.u >> 1);
    float y = v.f;
    y = y * (1.5f - 0.5f * x * y * y);
    y = y * (1.5f - 0.5f * x * y * y);
    y = y * (1.5f - 0.5f * x * y * y);
    return y;
}

static inline int32_t bf16_to_q20(uint16_t b) {
    uint32_t u = ((uint32_t)(b & 0x7FFFu)) << 16;
    if (u == 0) return 0;
    int32_t e = (int32_t)((u >> 23) & 0xFF) - 127;
    uint32_t m = (u & 0x7FFFFFu) | 0x800000u;              // m * 2^(e - 23); Q20 = m * 2^(e - 3)
    int32_t q;
    if (e - 3 >= 7) {
        q = 0x3FFFFFFF;                                    // saturate at ~1024.0
    } else if (e - 3 >= 0) {
        q = (int32_t)(m << (e - 3));
    } else {
        int32_t sh = 3 - e;
        q = sh >= 24 ? 0 : (int32_t)(m >> sh);
    }
    return (b & 0x8000u) ? -q : q;
}

static inline uint16_t q20_to_bf16(uint32_t q) {
    uint32_t u = from_q20(q);
    return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

static inline uint32_t exp_neg_q20(int32_t x_q20) {        // x_q20 <= 0 -> exp(x) in Q20 [0, 2^20]
    if (x_q20 >= 0) return 1048576u;
    if (x_q20 < -(20 << 20)) return 0u;
    // t = x * log2(e) in Q20, log2(e) * 2^20 = 1512775
    int32_t t = (int32_t)(((int64_t)x_q20 * 1512775LL) >> 20);
    int32_t k = t >> 20;                                   // k <= 0
    uint32_t f = (uint32_t)(t & 0xFFFFF);                  // f in [0, 2^20)
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
    const uint32_t mm4_addr   = get_arg_val<uint32_t>(0);
    const uint32_t ms_addr    = get_arg_val<uint32_t>(1);
    const uint32_t scale_addr = get_arg_val<uint32_t>(2);
    const uint32_t base_addr  = get_arg_val<uint32_t>(3);
    const uint32_t pre_addr   = get_arg_val<uint32_t>(4);
    const uint32_t post_addr  = get_arg_val<uint32_t>(5);
    const uint32_t comb_addr  = get_arg_val<uint32_t>(6);
    const uint32_t iters      = get_arg_val<uint32_t>(7);
    const uint32_t ms_pages   = get_arg_val<uint32_t>(8);
    const uint32_t s_rows     = get_arg_val<uint32_t>(9);
    constexpr uint32_t PAGE = 2048, cb_in = 0, cb_out = 16, EPS_Q = 1;
    constexpr auto mm4_args   = TensorAccessorArgs<0>();
    constexpr auto ms_args    = TensorAccessorArgs<mm4_args.next_compile_time_args_offset()>();
    constexpr auto scale_args = TensorAccessorArgs<ms_args.next_compile_time_args_offset()>();
    constexpr auto base_args  = TensorAccessorArgs<scale_args.next_compile_time_args_offset()>();
    constexpr auto pre_args   = TensorAccessorArgs<base_args.next_compile_time_args_offset()>();
    constexpr auto post_args  = TensorAccessorArgs<pre_args.next_compile_time_args_offset()>();
    constexpr auto comb_args  = TensorAccessorArgs<post_args.next_compile_time_args_offset()>();
    const auto mm4a   = TensorAccessor(mm4_args, mm4_addr, PAGE);
    const auto msa    = TensorAccessor(ms_args, ms_addr, PAGE);
    const auto scalea = TensorAccessor(scale_args, scale_addr, PAGE);
    const auto basea  = TensorAccessor(base_args, base_addr, PAGE);
    const auto prea   = TensorAccessor(pre_args, pre_addr, PAGE);
    const auto posta  = TensorAccessor(post_args, post_addr, PAGE);
    const auto comba  = TensorAccessor(comb_args, comb_addr, PAGE);

    cb_reserve_back(cb_in, 96);
    const uint32_t ib = get_write_ptr(cb_in);
    const uint32_t ob_pre  = ib + 86 * PAGE; // 4 pages
    const uint32_t ob_post = ib + 90 * PAGE; // 4 pages
    const uint32_t ob_comb = ib + 94 * PAGE; // 1 page

    for (uint32_t k = 0; k < 4; ++k) {
        noc_async_read_page(k, mm4a, ib + k * PAGE);
    }
    noc_async_read_page(0, scalea, ib + 4 * PAGE);
    noc_async_read_page(0, basea,  ib + 5 * PAGE);
    for (uint32_t k = 0; k < ms_pages; ++k) {
        noc_async_read_page(k, msa, ib + (6 + k) * PAGE);
    }

    volatile tt_l1_ptr uint32_t* z32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ob_pre);
    for (uint32_t i = 0; i < 9u * 512u; ++i) z32[i] = 0;
    noc_async_read_barrier();

    volatile tt_l1_ptr uint16_t* mm0   = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ib + 0 * PAGE);
    volatile tt_l1_ptr uint16_t* mm1   = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ib + 1 * PAGE);
    volatile tt_l1_ptr uint16_t* mm2   = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ib + 2 * PAGE);
    volatile tt_l1_ptr uint16_t* mm3   = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ib + 3 * PAGE);
    volatile tt_l1_ptr uint16_t* sc_p  = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ib + 4 * PAGE);
    volatile tt_l1_ptr uint16_t* bs_p  = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ib + 5 * PAGE);
    volatile tt_l1_ptr uint16_t* comb16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ob_comb);

    for (uint32_t r_tok = 0; r_tok < s_rows; ++r_tok) {
        float ms_val = 0.0f;
        if (ms_pages == 1) {
            volatile tt_l1_ptr uint16_t* ms_p = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ib + 6 * PAGE);
            ms_val = bf16_to_f32(ms_p[r_tok * 16u]);
        } else {
            float sum_sq = 0.0f;
            for (uint32_t k = 0; k < ms_pages; ++k) {
                volatile tt_l1_ptr uint32_t* p32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ib + (6 + k) * PAGE);
                union { uint32_t u; float f; } u2f = { p32[r_tok] };
                sum_sq += u2f.f;
            }
            ms_val = bf16_to_f32(f32_to_bf16(sum_sq * (1.0f / 20480.0f)));
        }
        const float inv_rms = bf16_to_f32(f32_to_bf16(fast_rsqrt(ms_val)));
        int32_t v[24];
        const uint32_t r_off = r_tok * 16u;
        for (uint32_t c = 0; c < 24; ++c) {
            const uint32_t sc_off = (c < 16) ? c : (256u + (c - 16));
            const uint32_t off = r_off + sc_off;
            const float s4 = (bf16_to_f32(mm0[off]) + bf16_to_f32(mm1[off])) + (bf16_to_f32(mm2[off]) + bf16_to_f32(mm3[off]));
            const float m_norm = bf16_to_f32(f32_to_bf16(bf16_to_f32(f32_to_bf16(s4)) * inv_rms));
            const float sc = bf16_to_f32(sc_p[sc_off]);
            const float bs = bf16_to_f32(bs_p[sc_off]);
            const uint16_t m_bf16 = f32_to_bf16(bf16_to_f32(f32_to_bf16(m_norm * sc)) + bs);
            v[c] = bf16_to_q20(m_bf16);
        }

        for (uint32_t k = 0; k < 4; ++k) {
            volatile tt_l1_ptr uint16_t* pre16  = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ob_pre  + k * PAGE);
            volatile tt_l1_ptr uint16_t* post16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ob_post + k * PAGE);

            uint32_t sq0 = sigmoid_q20(v[k]);
            uint16_t sb0 = q20_to_bf16(sq0);
            uint32_t sq0_round = to_q20(((uint32_t)sb0) << 16);
            pre16[r_off] = (sq0_round == 0) ? 0x3586u : q20_to_bf16(sq0_round + EPS_Q);

            uint32_t sq1 = sigmoid_q20(v[4 + k]);
            uint16_t sb1 = q20_to_bf16(sq1);
            uint32_t sq1_round = to_q20(((uint32_t)sb1) << 16);
            post16[r_off] = q20_to_bf16(sq1_round << 1);
        }

        uint32_t m[4][4];
        for (uint32_t r = 0; r < 4; ++r) {
            int32_t row_max = v[8 + r * 4];
            for (uint32_t c = 1; c < 4; ++c) {
                if (v[8 + r * 4 + c] > row_max) row_max = v[8 + r * 4 + c];
            }
            uint32_t e[4], sum_e = 0;
            for (uint32_t c = 0; c < 4; ++c) {
                e[c] = exp_neg_q20(v[8 + r * 4 + c] - row_max);
                sum_e += e[c];
            }
            uint64_t recip = (1ULL << 40) / (sum_e ? sum_e : 1u);
            for (uint32_t c = 0; c < 4; ++c) {
                uint32_t sm_q = qdiv(e[c], recip);
                uint16_t sm_b = q20_to_bf16(sm_q);
                uint32_t sm_round = to_q20(((uint32_t)sm_b) << 16);
                uint16_t sm_eps_b = (sm_round == 0) ? 0x3586u : q20_to_bf16(sm_round + EPS_Q);
                m[r][c] = to_q20(((uint32_t)sm_eps_b) << 16);
            }
        }

        for (uint32_t it = 0; it < 2 * iters - 1; ++it) {
            const bool cols = (it & 1) == 0;
            for (uint32_t a = 0; a < 4; ++a) {
                uint32_t s = EPS_Q;
                for (uint32_t b = 0; b < 4; ++b) s += cols ? m[b][a] : m[a][b];
                const uint64_t recip = (1ULL << 40) / s;
                for (uint32_t b = 0; b < 4; ++b) {
                    if (cols) m[b][a] = qdiv(m[b][a], recip); else m[a][b] = qdiv(m[a][b], recip);
                }
            }
        }

        const uint32_t comb_base = r_tok * 64u;
        for (uint32_t r = 0; r < 4; ++r) {
            for (uint32_t c = 0; c < 4; ++c) {
                comb16[comb_base + c * 16 + r] = q20_to_bf16(m[r][c]);
            }
        }
    }

    for (uint32_t k = 0; k < 4; ++k) {
        noc_async_write_page(k, prea,  ob_pre  + k * PAGE);
        noc_async_write_page(k, posta, ob_post + k * PAGE);
    }
    noc_async_write_page(0, comba, ob_comb);
    noc_async_write_barrier();
    (void)cb_out;
}
#else
void kernel_main() {
    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t iters = get_arg_val<uint32_t>(2);
    const uint32_t n = get_arg_val<uint32_t>(3);          // number of [4,4] matrices = tiles (pages), processed in turn
    constexpr uint32_t PAGE = 4096, cb_in = 0, cb_out = 16, EPS_Q = 1;   // 1e-6 * 2^20 ~= 1
    constexpr auto i_args = TensorAccessorArgs<0>();
    constexpr auto o_args = TensorAccessorArgs<i_args.next_compile_time_args_offset()>();
    const auto ia = TensorAccessor(i_args, in_addr, PAGE);
    const auto oa = TensorAccessor(o_args, out_addr, PAGE);

    cb_reserve_back(cb_in, 1);
    const uint32_t ib = get_write_ptr(cb_in);
    cb_reserve_back(cb_out, 1);
    const uint32_t ob = get_write_ptr(cb_out);
    for (uint32_t page = 0; page < n; ++page) {
    noc_async_read_page(page, ia, ib);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* in = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ib);

    uint32_t m[4][4];
    for (uint32_t r = 0; r < 4; ++r)
        for (uint32_t c = 0; c < 4; ++c) m[r][c] = to_q20(in[r * 16 + c]);

    for (uint32_t it = 0; it < 2 * iters - 1; ++it) {
        const bool cols = (it & 1) == 0;                // starts with a COLUMN normalization
        for (uint32_t a = 0; a < 4; ++a) {
            uint32_t s = EPS_Q;
            for (uint32_t b = 0; b < 4; ++b) s += cols ? m[b][a] : m[a][b];
            const uint32_t recip = (uint32_t)(((uint64_t)1 << 40) / s);
            for (uint32_t b = 0; b < 4; ++b) {
                if (cols) m[b][a] = qdiv(m[b][a], recip); else m[a][b] = qdiv(m[a][b], recip);
            }
        }
    }

    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ob);
    for (uint32_t i = 0; i < 1024; ++i) out[i] = 0;
    for (uint32_t r = 0; r < 4; ++r)
        for (uint32_t c = 0; c < 4; ++c) out[r * 16 + c] = from_q20(m[r][c]);
    noc_async_write_page(page, oa, ob);
    noc_async_write_barrier();
    }
}
#endif
