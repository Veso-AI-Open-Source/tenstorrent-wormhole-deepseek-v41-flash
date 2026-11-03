// SPDX-License-Identifier: Apache-2.0
// Fused 40-core x 2-RISC (80 workers, 2 col tiles/worker) s=1 mHC stream kernels on Wormhole B0.
// Uses hardware integer Q16 arithmetic for normal channels (|x| < 512) and exact IEEE-754 float32
// for massive-activation channels (|x| >= 512, 1-2 channels out of 5120).
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

static inline uint32_t f32_to_bits(float f) {
    union { float f; uint32_t u; } v = { f };
    return v.u;
}

static inline float q24_u64_to_f32(uint64_t v) {
    if (v == 0) return 0.0f;
    uint32_t lz = (uint32_t)__builtin_clzll(v);
    uint32_t p = 63u - lz; // MSB at bit p, value = v * 2^-24
    uint32_t man = (p >= 23u) ? (uint32_t)(v >> (p - 23u)) : (uint32_t)(v << (23u - p));
    uint32_t exp = p + 103u; // p - 24 + 127
    union { uint32_t u; float f; } u2f = { (exp << 23) | (man & 0x7FFFFFu) };
    return u2f.f;
}

static inline int32_t bf16_to_q16(uint16_t b) {
    uint32_t e = (b >> 7) & 0xFFu;
    if (e < 104u) return 0;
    uint32_t m = (b & 0x7Fu) | 0x80u; // value = m * 2^(e - 134); Q16 = m * 2^(e - 118)
    int32_t q = (e >= 118u) ? (int32_t)(m << (e - 118u)) : (int32_t)(m >> (118u - e));
    return (b & 0x8000u) ? -q : q;
}

static inline uint16_t q16_to_bf16(int32_t q) {
    if (q == 0) return 0;
    uint32_t sign = 0, u = (uint32_t)q;
    if (q < 0) { sign = 0x80000000u; u = 0u - u; }
    uint32_t p = 31u - (uint32_t)__builtin_clz(u);
    uint32_t man = (p >= 23u) ? (u >> (p - 23u)) : (u << (23u - p));
    uint32_t f32 = sign | ((p + 111u) << 23) | (man & 0x7FFFFFu);
    return (uint16_t)((f32 + 0x7FFFu + ((f32 >> 16) & 1u)) >> 16);
}

static inline int32_t q16_mul(int32_t a, int32_t b) {
    return (int32_t)(((int64_t)a * (int64_t)b) >> 16);
}

static inline int32_t q16_round_bf16(int32_t q) {
    return bf16_to_q16(q16_to_bf16(q));
}

void kernel_main() {
    constexpr uint32_t PAGE = 2048;
    constexpr uint32_t WT = 160; // 5120 / 32 = 160 tiles per HC copy
#if defined(IS_WRITER)
    constexpr uint32_t cb_id = 1;
    constexpr uint32_t is_writer = 1;
#else
    constexpr uint32_t cb_id = 0;
    constexpr uint32_t is_writer = 0;
#endif

#if defined(MODE_POST_PRE_MS)
    const uint32_t y_addr     = get_arg_val<uint32_t>(0);
    const uint32_t res_addr   = get_arg_val<uint32_t>(1);
    const uint32_t post_addr  = get_arg_val<uint32_t>(2);
    const uint32_t comb_addr  = get_arg_val<uint32_t>(3);
    const uint32_t pre_addr   = get_arg_val<uint32_t>(4);
    const uint32_t xout_addr  = get_arg_val<uint32_t>(5);
    const uint32_t hpre_addr  = get_arg_val<uint32_t>(6);
    const uint32_t ms_addr    = get_arg_val<uint32_t>(7);
    const uint32_t cid        = get_arg_val<uint32_t>(8); // 0..39
    const uint32_t s_rows     = get_arg_val<uint32_t>(9); // 1..8
    const uint32_t wid        = cid * 2u + is_writer;    // 0..79

    constexpr auto y_args    = TensorAccessorArgs<0>();
    constexpr auto res_args  = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    constexpr auto post_args = TensorAccessorArgs<res_args.next_compile_time_args_offset()>();
    constexpr auto comb_args = TensorAccessorArgs<post_args.next_compile_time_args_offset()>();
    constexpr auto pre_args  = TensorAccessorArgs<comb_args.next_compile_time_args_offset()>();
    constexpr auto xout_args = TensorAccessorArgs<pre_args.next_compile_time_args_offset()>();
    constexpr auto hpre_args = TensorAccessorArgs<xout_args.next_compile_time_args_offset()>();
    constexpr auto ms_args   = TensorAccessorArgs<hpre_args.next_compile_time_args_offset()>();

    const auto y_ac    = TensorAccessor(y_args, y_addr, PAGE);
    const auto res_ac  = TensorAccessor(res_args, res_addr, PAGE);
    const auto post_ac = TensorAccessor(post_args, post_addr, PAGE);
    const auto comb_ac = TensorAccessor(comb_args, comb_addr, PAGE);
    const auto pre_ac  = TensorAccessor(pre_args, pre_addr, PAGE);
    const auto xout_ac = TensorAccessor(xout_args, xout_addr, PAGE);
    const auto hpre_ac = TensorAccessor(hpre_args, hpre_addr, PAGE);
    const auto ms_ac   = TensorAccessor(ms_args, ms_addr, PAGE);

    cb_reserve_back(cb_id, 16);
    const uint32_t base_l1  = get_write_ptr(cb_id);
    const uint32_t buf_post = base_l1;                  // 4 pages (0..3)
    const uint32_t buf_pre  = base_l1 + 4 * PAGE;       // 4 pages (4..7)
    const uint32_t buf_comb = base_l1 + 8 * PAGE;       // 1 page  (8)
    const uint32_t buf_y    = base_l1 + 9 * PAGE;       // 1 page  (9)
    const uint32_t buf_res  = base_l1 + 10 * PAGE;      // 4 pages (10..13)
    const uint32_t buf_hpre = base_l1 + 14 * PAGE;      // 1 page  (14)
    const uint32_t buf_ms   = base_l1 + 15 * PAGE;      // 1 page  (15)

    for (uint32_t j = 0; j < 4; ++j) {
        noc_async_read_page(j, post_ac, buf_post + j * PAGE);
        noc_async_read_page(j, pre_ac,  buf_pre  + j * PAGE);
    }
    noc_async_read_page(0, comb_ac, buf_comb);
    noc_async_read_barrier();

    volatile tt_l1_ptr uint16_t* comb16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_comb);
    volatile tt_l1_ptr uint32_t* hpre32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf_hpre);
    for (uint32_t i = 0; i < 512; ++i) hpre32[i] = 0;

    uint64_t sum_sq_q24[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float big_sq[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    bool has_big[8] = {false, false, false, false, false, false, false, false};
    const uint32_t c_start = wid * 2u;
    for (uint32_t dc = 0; dc < 2; ++dc) {
        const uint32_t c = c_start + dc;
        noc_async_read_page(c, y_ac, buf_y);
        for (uint32_t i = 0; i < 4; ++i) {
            noc_async_read_page(i * WT + c, res_ac, buf_res + i * PAGE);
        }
        noc_async_read_barrier();

        volatile tt_l1_ptr uint16_t* y16   = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_y);
        volatile tt_l1_ptr uint16_t* r0_16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_res + 0 * PAGE);
        volatile tt_l1_ptr uint16_t* r1_16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_res + 1 * PAGE);
        volatile tt_l1_ptr uint16_t* r2_16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_res + 2 * PAGE);
        volatile tt_l1_ptr uint16_t* r3_16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_res + 3 * PAGE);
        volatile tt_l1_ptr uint16_t* hp16  = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_hpre);
        volatile tt_l1_ptr uint16_t* r_ptrs[4] = {r0_16, r1_16, r2_16, r3_16};

        for (uint32_t r = 0; r < s_rows; ++r) {
            const uint32_t r_off = r * 16u;
            const uint32_t c_base = r * 64u;
            int32_t post_v[4], pre_v[4], comb_v[4][4];
            for (uint32_t j = 0; j < 4; ++j) {
                volatile tt_l1_ptr uint16_t* p_post = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_post + j * PAGE);
                volatile tt_l1_ptr uint16_t* p_pre  = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_pre  + j * PAGE);
                post_v[j] = bf16_to_q16(p_post[r_off]);
                pre_v[j]  = bf16_to_q16(p_pre[r_off]);
                for (uint32_t i = 0; i < 4; ++i) {
                    comb_v[j][i] = bf16_to_q16(comb16[c_base + j * 16 + i]);
                }
            }
            for (uint32_t face = 0; face < 2; ++face) {
                const uint32_t base = face * 256u + r_off;
                for (uint32_t col = 0; col < 16; ++col) {
                    const uint32_t idx = base + col;
                    const uint16_t yb = y16[idx], b0 = r0_16[idx], b1 = r1_16[idx], b2 = r2_16[idx], b3 = r3_16[idx];
                    if (__builtin_expect(((yb & 0x7FFFu) >= 0x4400u) | ((b0 & 0x7FFFu) >= 0x4400u) | ((b1 & 0x7FFFu) >= 0x4400u) | ((b2 & 0x7FFFu) >= 0x4400u) | ((b3 & 0x7FFFu) >= 0x4400u), 0)) {
                        has_big[r] = true;
                        const float yf = bf16_to_f32(yb);
                        const float rf0 = bf16_to_f32(b0), rf1 = bf16_to_f32(b1), rf2 = bf16_to_f32(b2), rf3 = bf16_to_f32(b3);
                        float hp_f = 0.0f;
                        for (uint32_t j = 0; j < 4; ++j) {
                            volatile tt_l1_ptr uint16_t* p_post = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_post + j * PAGE);
                            volatile tt_l1_ptr uint16_t* p_pre  = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_pre  + j * PAGE);
                            const float mixed = bf16_to_f32(comb16[c_base + j * 16 + 0]) * rf0
                                              + bf16_to_f32(comb16[c_base + j * 16 + 1]) * rf1
                                              + bf16_to_f32(comb16[c_base + j * 16 + 2]) * rf2
                                              + bf16_to_f32(comb16[c_base + j * 16 + 3]) * rf3;
                            const uint16_t x_b = f32_to_bf16(bf16_to_f32(p_post[r_off]) * yf + mixed);
                            r_ptrs[j][idx] = x_b;
                            const float xf = bf16_to_f32(x_b);
                            hp_f += bf16_to_f32(f32_to_bf16(xf * bf16_to_f32(p_pre[r_off])));
                            big_sq[r] += xf * xf;
                        }
                        hp16[idx] = f32_to_bf16(hp_f);
                    } else {
                        const int32_t y_q = bf16_to_q16(yb);
                        const int32_t r0  = bf16_to_q16(b0);
                        const int32_t r1  = bf16_to_q16(b1);
                        const int32_t r2  = bf16_to_q16(b2);
                        const int32_t r3  = bf16_to_q16(b3);

                        int32_t hp_acc = 0;
                        for (uint32_t j = 0; j < 4; ++j) {
                            const int64_t mixed64 = (int64_t)comb_v[j][0] * r0 + (int64_t)comb_v[j][1] * r1
                                                  + (int64_t)comb_v[j][2] * r2 + (int64_t)comb_v[j][3] * r3
                                                  + (int64_t)post_v[j] * y_q;
                            const uint16_t x_b = q16_to_bf16((int32_t)(mixed64 >> 16));
                            r_ptrs[j][idx] = x_b;
                            const int32_t x_q = bf16_to_q16(x_b);
                            hp_acc += q16_round_bf16(q16_mul(x_q, pre_v[j]));
                            sum_sq_q24[r] += (uint64_t)(((int64_t)x_q * (int64_t)x_q) >> 8);
                        }
                        hp16[idx] = q16_to_bf16(hp_acc);
                    }
                }
            }
        }

        for (uint32_t j = 0; j < 4; ++j) {
            noc_async_write_page(j * WT + c, xout_ac, buf_res + j * PAGE);
        }
        noc_async_write_page(c, hpre_ac, buf_hpre);
        noc_async_write_barrier();
    }

    volatile tt_l1_ptr uint32_t* ms32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf_ms);
    for (uint32_t r = 0; r < s_rows; ++r) {
        float total_worker_sq = q24_u64_to_f32(sum_sq_q24[r]);
        if (has_big[r]) total_worker_sq += big_sq[r];
        ms32[r] = f32_to_bits(total_worker_sq);
    }
    noc_async_write_page(wid, ms_ac, buf_ms);
    noc_async_write_barrier();

#elif defined(MODE_POST_ONLY)
    const uint32_t y_addr     = get_arg_val<uint32_t>(0);
    const uint32_t res_addr   = get_arg_val<uint32_t>(1);
    const uint32_t post_addr  = get_arg_val<uint32_t>(2);
    const uint32_t comb_addr  = get_arg_val<uint32_t>(3);
    const uint32_t xout_addr  = get_arg_val<uint32_t>(4);
    const uint32_t cid        = get_arg_val<uint32_t>(5); // 0..39
    const uint32_t s_rows     = get_arg_val<uint32_t>(6); // 1..8
    const uint32_t wid        = cid * 2u + is_writer;    // 0..79

    constexpr auto y_args    = TensorAccessorArgs<0>();
    constexpr auto res_args  = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    constexpr auto post_args = TensorAccessorArgs<res_args.next_compile_time_args_offset()>();
    constexpr auto comb_args = TensorAccessorArgs<post_args.next_compile_time_args_offset()>();
    constexpr auto xout_args = TensorAccessorArgs<comb_args.next_compile_time_args_offset()>();

    const auto y_ac    = TensorAccessor(y_args, y_addr, PAGE);
    const auto res_ac  = TensorAccessor(res_args, res_addr, PAGE);
    const auto post_ac = TensorAccessor(post_args, post_addr, PAGE);
    const auto comb_ac = TensorAccessor(comb_args, comb_addr, PAGE);
    const auto xout_ac = TensorAccessor(xout_args, xout_addr, PAGE);

    cb_reserve_back(cb_id, 10);
    const uint32_t base_l1  = get_write_ptr(cb_id);
    const uint32_t buf_post = base_l1;                  // 4 pages (0..3)
    const uint32_t buf_comb = base_l1 + 4 * PAGE;       // 1 page  (4)
    const uint32_t buf_y    = base_l1 + 5 * PAGE;       // 1 page  (5)
    const uint32_t buf_res  = base_l1 + 6 * PAGE;       // 4 pages (6..9)

    for (uint32_t j = 0; j < 4; ++j) {
        noc_async_read_page(j, post_ac, buf_post + j * PAGE);
    }
    noc_async_read_page(0, comb_ac, buf_comb);
    noc_async_read_barrier();

    volatile tt_l1_ptr uint16_t* comb16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_comb);
    const uint32_t c_start = wid * 2u;
    for (uint32_t dc = 0; dc < 2; ++dc) {
        const uint32_t c = c_start + dc;
        noc_async_read_page(c, y_ac, buf_y);
        for (uint32_t i = 0; i < 4; ++i) {
            noc_async_read_page(i * WT + c, res_ac, buf_res + i * PAGE);
        }
        noc_async_read_barrier();

        volatile tt_l1_ptr uint16_t* y16   = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_y);
        volatile tt_l1_ptr uint16_t* r0_16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_res + 0 * PAGE);
        volatile tt_l1_ptr uint16_t* r1_16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_res + 1 * PAGE);
        volatile tt_l1_ptr uint16_t* r2_16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_res + 2 * PAGE);
        volatile tt_l1_ptr uint16_t* r3_16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_res + 3 * PAGE);
        volatile tt_l1_ptr uint16_t* r_ptrs[4] = {r0_16, r1_16, r2_16, r3_16};

        for (uint32_t r = 0; r < s_rows; ++r) {
            const uint32_t r_off = r * 16u;
            const uint32_t c_base = r * 64u;
            int32_t post_v[4], comb_v[4][4];
            for (uint32_t j = 0; j < 4; ++j) {
                volatile tt_l1_ptr uint16_t* p_post = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_post + j * PAGE);
                post_v[j] = bf16_to_q16(p_post[r_off]);
                for (uint32_t i = 0; i < 4; ++i) {
                    comb_v[j][i] = bf16_to_q16(comb16[c_base + j * 16 + i]);
                }
            }
            for (uint32_t face = 0; face < 2; ++face) {
                const uint32_t base = face * 256u + r_off;
                for (uint32_t col = 0; col < 16; ++col) {
                    const uint32_t idx = base + col;
                    const uint16_t yb = y16[idx], b0 = r0_16[idx], b1 = r1_16[idx], b2 = r2_16[idx], b3 = r3_16[idx];
                    if (__builtin_expect(((yb & 0x7FFFu) >= 0x4400u) | ((b0 & 0x7FFFu) >= 0x4400u) | ((b1 & 0x7FFFu) >= 0x4400u) | ((b2 & 0x7FFFu) >= 0x4400u) | ((b3 & 0x7FFFu) >= 0x4400u), 0)) {
                        const float yf = bf16_to_f32(yb);
                        const float rf0 = bf16_to_f32(b0), rf1 = bf16_to_f32(b1), rf2 = bf16_to_f32(b2), rf3 = bf16_to_f32(b3);
                        for (uint32_t j = 0; j < 4; ++j) {
                            volatile tt_l1_ptr uint16_t* p_post = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_post + j * PAGE);
                            const float mixed = bf16_to_f32(comb16[c_base + j * 16 + 0]) * rf0
                                              + bf16_to_f32(comb16[c_base + j * 16 + 1]) * rf1
                                              + bf16_to_f32(comb16[c_base + j * 16 + 2]) * rf2
                                              + bf16_to_f32(comb16[c_base + j * 16 + 3]) * rf3;
                            r_ptrs[j][idx] = f32_to_bf16(bf16_to_f32(p_post[r_off]) * yf + mixed);
                        }
                    } else {
                        const int32_t y_q = bf16_to_q16(yb);
                        const int32_t r0  = bf16_to_q16(b0);
                        const int32_t r1  = bf16_to_q16(b1);
                        const int32_t r2  = bf16_to_q16(b2);
                        const int32_t r3  = bf16_to_q16(b3);

                        for (uint32_t j = 0; j < 4; ++j) {
                            const int64_t mixed64 = (int64_t)comb_v[j][0] * r0 + (int64_t)comb_v[j][1] * r1
                                                  + (int64_t)comb_v[j][2] * r2 + (int64_t)comb_v[j][3] * r3
                                                  + (int64_t)post_v[j] * y_q;
                            r_ptrs[j][idx] = q16_to_bf16((int32_t)(mixed64 >> 16));
                        }
                    }
                }
            }
        }

        for (uint32_t j = 0; j < 4; ++j) {
            noc_async_write_page(j * WT + c, xout_ac, buf_res + j * PAGE);
        }
        noc_async_write_barrier();
    }
#else
    // MODE_PRE_MS: Reads x [1, 4, s, 5120] and pre [1, 4, s, 1] -> writes h_pre [1, 1, s, 5120] and ms_buf[wid]
    const uint32_t x_addr    = get_arg_val<uint32_t>(0);
    const uint32_t pre_addr  = get_arg_val<uint32_t>(1);
    const uint32_t hpre_addr = get_arg_val<uint32_t>(2);
    const uint32_t ms_addr   = get_arg_val<uint32_t>(3);
    const uint32_t cid       = get_arg_val<uint32_t>(4); // 0..39
    const uint32_t s_rows    = get_arg_val<uint32_t>(5); // 1..8
    const uint32_t wid       = cid * 2u + is_writer;    // 0..79

    constexpr auto x_args    = TensorAccessorArgs<0>();
    constexpr auto pre_args  = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto hpre_args = TensorAccessorArgs<pre_args.next_compile_time_args_offset()>();
    constexpr auto ms_args   = TensorAccessorArgs<hpre_args.next_compile_time_args_offset()>();

    const auto x_ac    = TensorAccessor(x_args, x_addr, PAGE);
    const auto pre_ac  = TensorAccessor(pre_args, pre_addr, PAGE);
    const auto hpre_ac = TensorAccessor(hpre_args, hpre_addr, PAGE);
    const auto ms_ac   = TensorAccessor(ms_args, ms_addr, PAGE);

    cb_reserve_back(cb_id, 10);
    const uint32_t base_l1  = get_write_ptr(cb_id);
    const uint32_t buf_pre  = base_l1;                  // 4 pages (0..3)
    const uint32_t buf_x    = base_l1 + 4 * PAGE;       // 4 pages (4..7)
    const uint32_t buf_hpre = base_l1 + 8 * PAGE;       // 1 page  (8)
    const uint32_t buf_ms   = base_l1 + 9 * PAGE;       // 1 page  (9)

    for (uint32_t j = 0; j < 4; ++j) {
        noc_async_read_page(j, pre_ac, buf_pre + j * PAGE);
    }
    noc_async_read_barrier();

    volatile tt_l1_ptr uint32_t* hpre32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf_hpre);
    for (uint32_t i = 0; i < 512; ++i) hpre32[i] = 0;

    uint64_t sum_sq_q24[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float big_sq[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    bool has_big[8] = {false, false, false, false, false, false, false, false};
    const uint32_t c_start = wid * 2u;
    for (uint32_t dc = 0; dc < 2; ++dc) {
        const uint32_t c = c_start + dc;
        for (uint32_t j = 0; j < 4; ++j) {
            noc_async_read_page(j * WT + c, x_ac, buf_x + j * PAGE);
        }
        noc_async_read_barrier();

        volatile tt_l1_ptr uint16_t* x0_16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_x + 0 * PAGE);
        volatile tt_l1_ptr uint16_t* x1_16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_x + 1 * PAGE);
        volatile tt_l1_ptr uint16_t* x2_16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_x + 2 * PAGE);
        volatile tt_l1_ptr uint16_t* x3_16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_x + 3 * PAGE);
        volatile tt_l1_ptr uint16_t* hp16  = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_hpre);

        for (uint32_t r = 0; r < s_rows; ++r) {
            const uint32_t r_off = r * 16u;
            int32_t pre_v[4];
            for (uint32_t j = 0; j < 4; ++j) {
                volatile tt_l1_ptr uint16_t* p_pre = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_pre + j * PAGE);
                pre_v[j] = bf16_to_q16(p_pre[r_off]);
            }
            for (uint32_t face = 0; face < 2; ++face) {
                const uint32_t base = face * 256u + r_off;
                for (uint32_t col = 0; col < 16; ++col) {
                    const uint32_t idx = base + col;
                    const uint16_t b0 = x0_16[idx], b1 = x1_16[idx], b2 = x2_16[idx], b3 = x3_16[idx];
                    if (__builtin_expect(((b0 & 0x7FFFu) >= 0x4400u) | ((b1 & 0x7FFFu) >= 0x4400u) | ((b2 & 0x7FFFu) >= 0x4400u) | ((b3 & 0x7FFFu) >= 0x4400u), 0)) {
                        has_big[r] = true;
                        const float xf0 = bf16_to_f32(b0), xf1 = bf16_to_f32(b1), xf2 = bf16_to_f32(b2), xf3 = bf16_to_f32(b3);
                        big_sq[r] += (xf0 * xf0 + xf1 * xf1) + (xf2 * xf2 + xf3 * xf3);
                        float hp_f = 0.0f;
                        const float xfs[4] = {xf0, xf1, xf2, xf3};
                        for (uint32_t j = 0; j < 4; ++j) {
                            volatile tt_l1_ptr uint16_t* p_pre = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_pre + j * PAGE);
                            hp_f += bf16_to_f32(f32_to_bf16(xfs[j] * bf16_to_f32(p_pre[r_off])));
                        }
                        hp16[idx] = f32_to_bf16(hp_f);
                    } else {
                        const int32_t x0 = bf16_to_q16(b0);
                        const int32_t x1 = bf16_to_q16(b1);
                        const int32_t x2 = bf16_to_q16(b2);
                        const int32_t x3 = bf16_to_q16(b3);

                        sum_sq_q24[r] += (uint64_t)(((int64_t)x0 * x0 + (int64_t)x1 * x1 + (int64_t)x2 * x2 + (int64_t)x3 * x3) >> 8);
                        const int32_t hp = (q16_round_bf16(q16_mul(x0, pre_v[0])) + q16_round_bf16(q16_mul(x1, pre_v[1])))
                                         + (q16_round_bf16(q16_mul(x2, pre_v[2])) + q16_round_bf16(q16_mul(x3, pre_v[3])));
                        hp16[idx] = q16_to_bf16(hp);
                    }
                }
            }
        }

        noc_async_write_page(c, hpre_ac, buf_hpre);
        noc_async_write_barrier();
    }

    volatile tt_l1_ptr uint32_t* ms32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf_ms);
    for (uint32_t r = 0; r < s_rows; ++r) {
        float total_worker_sq = q24_u64_to_f32(sum_sq_q24[r]);
        if (has_big[r]) total_worker_sq += big_sq[r];
        ms32[r] = f32_to_bits(total_worker_sq);
    }
    noc_async_write_page(wid, ms_ac, buf_ms);
    noc_async_write_barrier();
#endif
}
