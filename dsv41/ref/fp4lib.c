// Host-side byte helpers for DeepSeek-V4.1-Flash: MXFP4 dequant and the tile packer, used by the expert store. Needs AVX-512.
//   dequant_fp4:    MXFP4 [rows, cols/2] nibbles (LOW first) + E8M0 [rows, cols/32] -> float32 [rows, cols]
//   bfp_roundtrip:  quantize float32 rows to a Tenstorrent-style block-float grid IN PLACE
//                   (16 values share the max exponent; `mant_bits` magnitude bits + sign), so the oracle
//                   can run with exactly the weights the device would see under bfp4 / bfp8.
// build: gcc -O3 -march=native -fopenmp -shared -fPIC -o libfp4.so fp4lib.c -lm
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

void dequant_fp4(const uint8_t *q, const uint8_t *s, float *w, long rows, long cols) {
  const __m128i m4 = _mm_set1_epi8(0x0F);
  // 2 * E2M1 value, so the table is integral; bit 3 of the nibble is the sign
  const __m128i lut = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
  long nblk = cols / 32;
  #pragma omp parallel for schedule(static)
  for (long r = 0; r < rows; r++) {
    const uint8_t *qr = q + r * (cols / 2), *sr = s + r * nblk;
    float *wr = w + r * cols;
    for (long b = 0; b < nblk; b++) {
      __m128i y = _mm_loadu_si128((const __m128i *)(qr + 16 * b));
      __m128i lo = _mm_shuffle_epi8(lut, _mm_and_si128(y, m4));
      __m128i hi = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(y, 4), m4));
      __m512 sc = _mm512_set1_ps(ldexpf(0.5f, (int)sr[b] - 127));
      _mm512_storeu_ps(wr + 32 * b, _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_unpacklo_epi8(lo, hi))), sc));
      _mm512_storeu_ps(wr + 32 * b + 16, _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_unpackhi_epi8(lo, hi))), sc));
    }
  }
}

// mode 0: truncate toward zero; mode 1: round half away from zero; mode 2: round half to even
void bfp_roundtrip(float *w, long rows, long cols, int mant_bits, int mode) {
  const float maxm = (float)((1 << mant_bits) - 1);
  #pragma omp parallel for schedule(static)
  for (long r = 0; r < rows; r++) {
    float *wr = w + r * cols;
    for (long b = 0; b + 16 <= cols; b += 16) {
      float amax = 0; for (int i = 0; i < 16; i++) { float a = fabsf(wr[b + i]); if (a > amax) amax = a; }
      if (amax == 0) continue;
      int e; frexpf(amax, &e);                           // amax = m * 2^e, m in [0.5, 1)
      float unit = ldexpf(1.0f, e - mant_bits);          // top mantissa bit sits at 2^(e-1)
      for (int i = 0; i < 16; i++) {
        float v = fabsf(wr[b + i]) / unit, qv;
        if (mode == 0) qv = floorf(v); else if (mode == 1) qv = floorf(v + 0.5f); else qv = nearbyintf(v);
        if (qv > maxm) qv = maxm;
        wr[b + i] = copysignf(qv * unit, wr[b + i]);
      }
    }
  }
}

// Same quantizer, but the 16-value blocks run DOWN the rows (w is [rows, cols], block = 16 consecutive
// rows of one column). That is the device's view when the engine stores the matrix transposed for
// x @ W: saves two 47 MB transposes per expert matrix in the quality test.
void bfp_roundtrip_t(float *w, long rows, long cols, int mant_bits, int mode) {
  const float maxm = (float)((1 << mant_bits) - 1);
  #pragma omp parallel for schedule(static)
  for (long rb = 0; rb < rows / 16; rb++) {
    float *base = w + rb * 16 * cols;
    for (long c = 0; c < cols; c++) {
      float amax = 0; for (int i = 0; i < 16; i++) { float a = fabsf(base[i * cols + c]); if (a > amax) amax = a; }
      if (amax == 0) continue;
      int e; frexpf(amax, &e);
      float unit = ldexpf(1.0f, e - mant_bits);
      for (int i = 0; i < 16; i++) {
        float x = base[i * cols + c], v = fabsf(x) / unit, qv;
        if (mode == 0) qv = floorf(v); else if (mode == 1) qv = floorf(v + 0.5f); else qv = nearbyintf(v);
        if (qv > maxm) qv = maxm;
        base[i * cols + c] = copysignf(qv * unit, x);
      }
    }
  }
}

// ---- ttnn block-float TILE packer (format decoded from ttnn.dump_tensor output, 2026-09-17) ----
// A 32x32 tile = 64 exponent bytes + data. Faces in order TL, TR, BL, BR, each 16x16 row-major. One
// exponent per 16-value face row = IEEE biased exponent of the block max (127 + floor(log2 amax)), 0
// for an all-zero block. Value = (-1)^s * mant * 2^(E - 127 - (mant_bits-1)), mant round-half-even and
// clamped. bfp8: 1 byte/value, sign in bit 7. bfp4: 2 values/byte, LOW nibble first, sign in bit 3.
// Packs the device matrix D[rows x cols] where D[i][j] = W[(j0 + j) * wcols + (i0 + i)], i.e. a
// sub-block of W TRANSPOSED: exactly the engine's x @ W layout, without materializing the transpose.
void pack_bfp_tiles_t(const float *W, long wcols, long i0, long j0, long rows, long cols, int mant_bits, uint8_t *out) {
  const long tr_n = rows / 32, tc_n = cols / 32;
  const long dbytes = mant_bits == 7 ? 1024 : 512, tbytes = 64 + dbytes;
  const float maxm = (float)((1 << mant_bits) - 1);
  #pragma omp parallel for schedule(static) collapse(2)
  for (long tr = 0; tr < tr_n; tr++)
    for (long tc = 0; tc < tc_n; tc++) {
      uint8_t *t = out + (tr * tc_n + tc) * tbytes, *data = t + 64;
      if (mant_bits != 7) memset(data, 0, dbytes);
      for (int f = 0; f < 4; f++)
        for (int fr = 0; fr < 16; fr++) {
          long i = tr * 32 + (f >> 1) * 16 + fr, jb = tc * 32 + (f & 1) * 16;
          float v[16], amax = 0;
          for (int k = 0; k < 16; k++) { v[k] = W[(j0 + jb + k) * wcols + (i0 + i)]; float a = fabsf(v[k]); if (a > amax) amax = a; }
          int slot = f * 16 + fr;
          if (amax == 0) { t[slot] = 0; if (mant_bits == 7) memset(data + slot * 16, 0, 16); continue; }
          int e; frexpf(amax, &e);                                  // amax = m * 2^e, m in [0.5,1)
          t[slot] = (uint8_t)(127 + e - 1);
          float inv = ldexpf(1.0f, mant_bits - e);                  // 1 / unit
          for (int k = 0; k < 16; k++) {
            float q = nearbyintf(fabsf(v[k]) * inv); if (q > maxm) q = maxm;
            uint8_t m = (uint8_t)q, s = v[k] < 0 && m ? 1 : 0;
            if (mant_bits == 7) data[slot * 16 + k] = (uint8_t)(m | (s << 7));
            else data[slot * 8 + (k >> 1)] |= (uint8_t)((m | (s << 3)) << ((k & 1) * 4));
          }
        }
    }
}

// Fused, L1-cache-local dequant_fp4 + 8-shard pack_bfp_tiles_t directly from raw (q, s) bytes.
// Eliminates the 47 MB intermediate float32 matrix and the 20,480-byte power-of-two strided reads.
// For is_w2 == 0 (w1/w3): W is [2304, 5120], 8 shards of D = W.T[:, 288*k : 288*(k+1)] ([5120, 288] = 160x9 tiles).
// For is_w2 == 1 (w2):    W is [5120, 2304], 8 shards of D = W.T[288*k : 288*(k+1), :] ([288, 5120] = 9x160 tiles).
void pack_expert_fp4_shards(const uint8_t *q, const uint8_t *s, int is_w2, int mant_bits, uint8_t *out) {
  const __m128i m4 = _mm_set1_epi8(0x0F);
  const __m128i lut = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
  const long wcols = is_w2 ? 2304 : 5120;
  const long q_stride = wcols / 2;
  const long s_stride = wcols / 32;
  const long tr_n = is_w2 ? 9 : 160;
  const long tc_n = is_w2 ? 160 : 9;
  const long dbytes = mant_bits == 7 ? 1024 : 512, tbytes = 64 + dbytes;
  const long shard_tiles = tr_n * tc_n; // 1440 tiles per shard
  const long total_tiles = 8 * shard_tiles; // 11520 tiles total
  const float maxm = (float)((1 << mant_bits) - 1);

  #pragma omp parallel for schedule(static)
  for (long tid = 0; tid < total_tiles; tid++) {
    long k_sh = tid / shard_tiles;
    long rem = tid % shard_tiles;
    long tr = rem / tc_n;
    long tc = rem % tc_n;

    // In W [wrows, wcols]:
    // - 32 rows r0 .. r0 + 31 correspond to the j dimension of D = W.T
    // - 1 MXFP4 32-element block b corresponds to the i dimension of D = W.T
    long r0 = is_w2 ? (tc * 32) : (k_sh * 288 + tc * 32);
    long b  = is_w2 ? (k_sh * 9 + tr) : tr;

    float blk[32][32] __attribute__((aligned(64)));
    for (int jr = 0; jr < 32; jr++) {
      long r = r0 + jr;
      __m128i y = _mm_loadu_si128((const __m128i *)(q + r * q_stride + 16 * b));
      __m128i lo = _mm_shuffle_epi8(lut, _mm_and_si128(y, m4));
      __m128i hi = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(y, 4), m4));
      __m512 sc = _mm512_set1_ps(ldexpf(0.5f, (int)s[r * s_stride + b] - 127));
      _mm512_store_ps(&blk[jr][0],  _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_unpacklo_epi8(lo, hi))), sc));
      _mm512_store_ps(&blk[jr][16], _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_unpackhi_epi8(lo, hi))), sc));
    }

    uint8_t *t = out + tid * tbytes, *data = t + 64;
    if (mant_bits != 7) memset(data, 0, dbytes);
    for (int f = 0; f < 4; f++) {
      int jb = (f & 1) * 16;
      for (int fr = 0; fr < 16; fr++) {
        int ic = (f >> 1) * 16 + fr;
        float v[16], amax = 0;
        for (int k = 0; k < 16; k++) {
          float val = blk[jb + k][ic];
          v[k] = val;
          float a = fabsf(val);
          if (a > amax) amax = a;
        }
        int slot = f * 16 + fr;
        if (amax == 0) {
          t[slot] = 0;
          if (mant_bits == 7) memset(data + slot * 16, 0, 16);
          continue;
        }
        int e;
        frexpf(amax, &e);
        t[slot] = (uint8_t)(127 + e - 1);
        float inv = ldexpf(1.0f, mant_bits - e);
        for (int k = 0; k < 16; k++) {
          float qv = nearbyintf(fabsf(v[k]) * inv);
          if (qv > maxm) qv = maxm;
          uint8_t m = (uint8_t)qv, sg = v[k] < 0 && m ? 1 : 0;
          if (mant_bits == 7) data[slot * 16 + k] = (uint8_t)(m | (sg << 7));
          else data[slot * 8 + (k >> 1)] |= (uint8_t)((m | (sg << 3)) << ((k & 1) * 4));
        }
      }
    }
  }
}
