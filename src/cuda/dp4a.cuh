// Integer dot products of the routed-expert formats against activations quantized to q8_1 (int8 per 32 values),
// transcribed from llama.cpp's ggml-cuda/vecdotq.cuh at 3cf0325 (MIT): codebook bytes with their signs applied by
// __vcmpne4/__vsub4, four products per dp4a. dot32<T>(row, sb, xq) covers the 32 values of sub-block sb of a row.
#pragma once

#include "dequant.cuh"

namespace bl::cuda {

struct BlockQ8 {        // block_q8_1: d and d * sum(qs) as half2, 32 int8
    half2  ds;
    int8_t qs[32];
};

__device__ __forceinline__ int ld_b2(const void * x, int i32) {   // 4 bytes from a 2-byte aligned address
    const uint16_t * x16 = static_cast<const uint16_t *>(x);
    return x16[2 * i32] | (x16[2 * i32 + 1] << 16);
}
__device__ __forceinline__ int ld_b4(const void * x, int i32) { return static_cast<const int *>(x)[i32]; }

__device__ __forceinline__ uint32_t unpack_ksigns(uint8_t v) {
    const uint32_t p = __popc(v) & 1;   // the 8th sign is the parity of the 7 stored ones
    return (v ^ p << 7) * 0x01010101u;
}

__device__ __forceinline__ int2 table16(int q4, const int8_t * table) {   // 8 nibbles -> 8 table bytes (two ints)
    const uint32_t * t32 = reinterpret_cast<const uint32_t *>(table);
    uint32_t tmp[2];
    const uint32_t sel = 0x32103210 | ((q4 & 0x88888888) >> 1);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint32_t sh = 16 * i;
        const uint32_t lo = __byte_perm(t32[0], t32[1], q4 >> sh);
        const uint32_t hi = __byte_perm(t32[2], t32[3], q4 >> sh);
        tmp[i] = __byte_perm(lo, hi, sel >> sh);
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
}

template <int T> __device__ __forceinline__ float dot32(const uint8_t * row, int sb, const BlockQ8 * xq);

template <> __device__ __forceinline__ float dot32<IQ2_XXS>(const uint8_t * row, int sb, const BlockQ8 * xq) {
    const block_iq2_xxs * b = reinterpret_cast<const block_iq2_xxs *>(row) + sb / 8;
    const int iqs = 2 * (sb % 8);
    const BlockQ8 & q = xq[sb];
    const int q2 = ld_b2(b->qs, iqs);
    const uint8_t * aux8 = reinterpret_cast<const uint8_t *>(&q2);
    const uint32_t aux32 = ld_b2(b->qs, iqs + 1);
    int sumi = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const uint2 grid = reinterpret_cast<const uint2 *>(iq2xxs_grid)[aux8[k0 / 2]];
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));
        const int s0 = __vcmpne4(signs & 0x08040201, 0), s1 = __vcmpne4(signs & 0x80402010, 0);
        sumi = __dp4a(static_cast<int>(__vsub4(grid.x ^ s0, s0)), ld_b4(q.qs, k0), sumi);
        sumi = __dp4a(static_cast<int>(__vsub4(grid.y ^ s1, s1)), ld_b4(q.qs, k0 + 1), sumi);
    }
    const int ls = aux32 >> 27 | 1;
    sumi = sumi * ls / 8;
    return __half2float(b->d) * __low2float(q.ds) * sumi;
}

template <> __device__ __forceinline__ float dot32<IQ2_XS>(const uint8_t * row, int sb, const BlockQ8 * xq) {
    const block_iq2_xs * b = reinterpret_cast<const block_iq2_xs *>(row) + sb / 8;
    const int iqs = 2 * (sb % 8);
    const BlockQ8 & q = xq[sb];
    const int2 packed = make_int2(ld_b2(b->qs, iqs), ld_b2(b->qs, iqs + 1));
    const uint16_t * q2 = reinterpret_cast<const uint16_t *>(&packed);
    const int ls0 = b->scales[iqs / 2] & 0x0F, ls1 = b->scales[iqs / 2] >> 4;
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint2 grid = reinterpret_cast<const uint2 *>(iq2xs_grid)[q2[l0 / 2] & 0x1FF];
        const uint32_t signs = unpack_ksigns(q2[l0 / 2] >> 9);
        const int s0 = __vcmpne4(signs & 0x08040201, 0), s1 = __vcmpne4(signs & 0x80402010, 0);
        const int gl = __vsub4(grid.x ^ s0, s0), gh = __vsub4(grid.y ^ s1, s1);
        if (l0 < 4) {
            sumi0 = __dp4a(gl, ld_b4(q.qs, l0), sumi0);
            sumi0 = __dp4a(gh, ld_b4(q.qs, l0 + 1), sumi0);
        } else {
            sumi1 = __dp4a(gl, ld_b4(q.qs, l0), sumi1);
            sumi1 = __dp4a(gh, ld_b4(q.qs, l0 + 1), sumi1);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    return __half2float(b->d) * __low2float(q.ds) * sumi;
}

template <> __device__ __forceinline__ float dot32<IQ2_S>(const uint8_t * row, int sb, const BlockQ8 * xq) {
    const block_iq2_s * b = reinterpret_cast<const block_iq2_s *>(row) + sb / 8;
    const int iqs = 2 * (sb % 8);
    const BlockQ8 & q = xq[sb];
    const int qs_packed = ld_b2(b->qs, iqs / 2);
    const uint8_t * qs = reinterpret_cast<const uint8_t *>(&qs_packed);
    const int qh = b->qh[iqs / 2];
    const int sp32 = ld_b2(b->qs, QK_K / 32 + iqs / 2);
    const uint8_t * sp8 = reinterpret_cast<const uint8_t *>(&sp32);
    const int ls0 = b->scales[iqs / 2] & 0x0F, ls1 = b->scales[iqs / 2] >> 4;
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int * grid = reinterpret_cast<const int *>(iq2s_grid + (qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)));
        const int s0 = __vcmpne4(((sp8[l0 / 2] & 0x03) << 7) | ((sp8[l0 / 2] & 0x0C) << 21), 0);
        const int s1 = __vcmpne4(((sp8[l0 / 2] & 0x30) << 3) | ((sp8[l0 / 2] & 0xC0) << 17), 0);
        const int gl = __vsub4(grid[0] ^ s0, s0), gh = __vsub4(grid[1] ^ s1, s1);
        if (l0 < 4) {
            sumi0 = __dp4a(gl, ld_b4(q.qs, l0), sumi0);
            sumi0 = __dp4a(gh, ld_b4(q.qs, l0 + 1), sumi0);
        } else {
            sumi1 = __dp4a(gl, ld_b4(q.qs, l0), sumi1);
            sumi1 = __dp4a(gh, ld_b4(q.qs, l0 + 1), sumi1);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    return __half2float(b->d) * __low2float(q.ds) * sumi;
}

template <> __device__ __forceinline__ float dot32<IQ3_XXS>(const uint8_t * row, int sb, const BlockQ8 * xq) {
    const block_iq3_xxs * b = reinterpret_cast<const block_iq3_xxs *>(row) + sb / 8;
    const int iqs = 2 * (sb % 8);
    const BlockQ8 & q = xq[sb];
    const int2 packed = make_int2(ld_b2(b->qs, iqs), ld_b2(b->qs, iqs + 1));
    const uint8_t * q3 = reinterpret_cast<const uint8_t *>(&packed);
    const uint32_t aux32 = ld_b2(b->qs, QK_K / 16 + iqs / 2);
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid = make_int2(iq3xxs_grid[q3[l0]], iq3xxs_grid[q3[l0 + 1]]);
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * l0 / 2));
        const int s0 = __vcmpne4(signs & 0x08040201, 0), s1 = __vcmpne4(signs & 0x80402010, 0);
        sumi = __dp4a(static_cast<int>(__vsub4(grid.x ^ s0, s0)), ld_b4(q.qs, l0), sumi);
        sumi = __dp4a(static_cast<int>(__vsub4(grid.y ^ s1, s1)), ld_b4(q.qs, l0 + 1), sumi);
    }
    const int ls = aux32 >> 28;
    sumi = (ls * sumi + sumi / 2) / 2;
    return __half2float(b->d) * __low2float(q.ds) * sumi;
}

template <> __device__ __forceinline__ float dot32<IQ3_S>(const uint8_t * row, int sb, const BlockQ8 * xq) {
    const block_iq3_s * b = reinterpret_cast<const block_iq3_s *>(row) + sb / 8;
    const int iqs = 2 * (sb % 8);
    const BlockQ8 & q = xq[sb];
    const int2 packed = make_int2(ld_b2(b->qs, iqs), ld_b2(b->qs, iqs + 1));
    const uint8_t * qs = reinterpret_cast<const uint8_t *>(&packed);
    const int qh = b->qh[iqs / 2];
    const int sp32 = ld_b2(b->signs, iqs / 2);
    const uint8_t * sp8 = reinterpret_cast<const uint8_t *>(&sp32);
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid = make_int2(iq3s_grid[qs[l0] | ((qh << (8 - l0)) & 0x100)],
                                    iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
        const int s0 = __vcmpne4(((sp8[l0 / 2] & 0x03) << 7) | ((sp8[l0 / 2] & 0x0C) << 21), 0);
        const int s1 = __vcmpne4(((sp8[l0 / 2] & 0x30) << 3) | ((sp8[l0 / 2] & 0xC0) << 17), 0);
        sumi = __dp4a(static_cast<int>(__vsub4(grid.x ^ s0, s0)), ld_b4(q.qs, l0), sumi);
        sumi = __dp4a(static_cast<int>(__vsub4(grid.y ^ s1, s1)), ld_b4(q.qs, l0 + 1), sumi);
    }
    sumi *= 1 + 2 * ((b->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
    return __half2float(b->d) * __low2float(q.ds) * sumi;
}

template <> __device__ __forceinline__ float dot32<IQ4_NL>(const uint8_t * row, int sb, const BlockQ8 * xq) {
    const block_iq4_nl * b = reinterpret_cast<const block_iq4_nl *>(row) + sb;   // 32 values per block
    const BlockQ8 & q = xq[sb];
    int sumi = 0;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const int2 v = table16(ld_b2(b->qs, l), reinterpret_cast<const int8_t *>(kvalues_iq4nl));
        sumi = __dp4a(v.x, ld_b4(q.qs, l), sumi);
        sumi = __dp4a(v.y, ld_b4(q.qs, l + 4), sumi);
    }
    return __half2float(b->d) * __low2float(q.ds) * sumi;
}

template <> __device__ __forceinline__ float dot32<Q2_0>(const uint8_t * row, int sb, const BlockQ8 * xq) {
    const block_q2_0 * b = reinterpret_cast<const block_q2_0 *>(row) + sb / 2;   // 64 values per block
    const int16_t * qs = reinterpret_cast<const int16_t *>(b->qs) + (sb % 2) * 4;
    const BlockQ8 & q = xq[sb];
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int v = qs[j];
        // crumbs 0..3 -> bytes -1, 0, 1, 2 (Q2_0 stores q and means q - 1)
        const int qe = __byte_perm(0x020100FF, 0x020100FF, v >> 0);
        const int qo = __byte_perm(0x020100FF, 0x020100FF, v >> 2);
        sumi = __dp4a(ld_b4(q.qs, 2 * j), static_cast<int>(__byte_perm(qe, qo, 0x5140)), sumi);
        sumi = __dp4a(ld_b4(q.qs, 2 * j + 1), static_cast<int>(__byte_perm(qe, qo, 0x7362)), sumi);
    }
    return __half2float(b->d) * __low2float(q.ds) * sumi;
}

// x[32*i .. 32*i+31] -> q8_1 block i, by one warp (ggml's quantize_q8_1: d = amax/127, q = round(x/d))
__device__ __forceinline__ void quantize_block_q8(const float * x, BlockQ8 * out, int lane) {
    const float v = x[lane];
    float amax = fabsf(v), sum = v;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    }
    const float d = amax / 127.f;
    out->qs[lane] = amax == 0.f ? 0 : static_cast<int8_t>(roundf(v / d));
    if (lane == 0) out->ds = make_half2(__float2half(d), __float2half(sum));
}

}  // namespace bl::cuda

namespace bl::cuda {

// Multi-token versions: a 32-value sub-block's codebook bytes and signs are decoded once, then dotted with the q8
// activations of up to NT tokens (xq + t*xs blocks). Each token's integer scale formula is applied exactly as in
// dot32, so a token's result does not depend on which other tokens share the sub-block.
template <int F, int NT> __device__ __forceinline__ void dot32m(const uint8_t * row, int sb, const BlockQ8 * xq, int xs, int nt,
                                                                float * acc, int tmask = -1);   // bit t: token t (all by default)

#define BL_TOK_LOOP for (int t = 0; t < NT; ++t) if (t < nt && ((tmask >> t) & 1))

template <int NT> __device__ __forceinline__ void dot32m_iq2xxs(const uint8_t * row, int sb, const BlockQ8 * xq, int xs, int nt,
                                                               float * acc, int tmask) {
    const block_iq2_xxs * b = reinterpret_cast<const block_iq2_xxs *>(row) + sb / 8;
    const int iqs = 2 * (sb % 8);
    const int q2 = ld_b2(b->qs, iqs);
    const uint8_t * aux8 = reinterpret_cast<const uint8_t *>(&q2);
    const uint32_t aux32 = ld_b2(b->qs, iqs + 1);
    int gv[8];
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const uint2 grid = reinterpret_cast<const uint2 *>(iq2xxs_grid)[aux8[k0 / 2]];
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));
        const int s0 = __vcmpne4(signs & 0x08040201, 0), s1 = __vcmpne4(signs & 0x80402010, 0);
        gv[k0] = __vsub4(grid.x ^ s0, s0);
        gv[k0 + 1] = __vsub4(grid.y ^ s1, s1);
    }
    const int ls = aux32 >> 27 | 1;
    const float d = __half2float(b->d);
    BL_TOK_LOOP {
        const BlockQ8 & q = xq[t * xs + sb];
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) sumi = __dp4a(gv[j], ld_b4(q.qs, j), sumi);
        sumi = sumi * ls / 8;
        acc[t] += d * __low2float(q.ds) * sumi;
    }
}

// shared shape of IQ2_XS / IQ2_S: two 16-value halves with their own 4-bit scales
template <int NT> __device__ __forceinline__ void finish_two_halves(const int * gv, int ls0, int ls1, float d, int sb,
                                                                    const BlockQ8 * xq, int xs, int nt, float * acc, int tmask) {
    BL_TOK_LOOP {
        const BlockQ8 & q = xq[t * xs + sb];
        int s0 = 0, s1 = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) s0 = __dp4a(gv[j], ld_b4(q.qs, j), s0);
#pragma unroll
        for (int j = 4; j < 8; ++j) s1 = __dp4a(gv[j], ld_b4(q.qs, j), s1);
        const int sumi = (s0 * ls0 + s1 * ls1 + (s0 + s1) / 2) / 4;
        acc[t] += d * __low2float(q.ds) * sumi;
    }
}

template <int NT> __device__ __forceinline__ void dot32m_iq2xs(const uint8_t * row, int sb, const BlockQ8 * xq, int xs, int nt,
                                                              float * acc, int tmask) {
    const block_iq2_xs * b = reinterpret_cast<const block_iq2_xs *>(row) + sb / 8;
    const int iqs = 2 * (sb % 8);
    const int2 packed = make_int2(ld_b2(b->qs, iqs), ld_b2(b->qs, iqs + 1));
    const uint16_t * q2 = reinterpret_cast<const uint16_t *>(&packed);
    int gv[8];
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint2 grid = reinterpret_cast<const uint2 *>(iq2xs_grid)[q2[l0 / 2] & 0x1FF];
        const uint32_t signs = unpack_ksigns(q2[l0 / 2] >> 9);
        const int s0 = __vcmpne4(signs & 0x08040201, 0), s1 = __vcmpne4(signs & 0x80402010, 0);
        gv[l0] = __vsub4(grid.x ^ s0, s0);
        gv[l0 + 1] = __vsub4(grid.y ^ s1, s1);
    }
    finish_two_halves<NT>(gv, b->scales[iqs / 2] & 0x0F, b->scales[iqs / 2] >> 4, __half2float(b->d), sb, xq, xs, nt, acc, tmask);
}

template <int NT> __device__ __forceinline__ void dot32m_iq2s(const uint8_t * row, int sb, const BlockQ8 * xq, int xs, int nt,
                                                             float * acc, int tmask) {
    const block_iq2_s * b = reinterpret_cast<const block_iq2_s *>(row) + sb / 8;
    const int iqs = 2 * (sb % 8);
    const int qs_packed = ld_b2(b->qs, iqs / 2);
    const uint8_t * qs = reinterpret_cast<const uint8_t *>(&qs_packed);
    const int qh = b->qh[iqs / 2];
    const int sp32 = ld_b2(b->qs, QK_K / 32 + iqs / 2);
    const uint8_t * sp8 = reinterpret_cast<const uint8_t *>(&sp32);
    int gv[8];
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int * grid = reinterpret_cast<const int *>(iq2s_grid + (qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)));
        const int s0 = __vcmpne4(((sp8[l0 / 2] & 0x03) << 7) | ((sp8[l0 / 2] & 0x0C) << 21), 0);
        const int s1 = __vcmpne4(((sp8[l0 / 2] & 0x30) << 3) | ((sp8[l0 / 2] & 0xC0) << 17), 0);
        gv[l0] = __vsub4(grid[0] ^ s0, s0);
        gv[l0 + 1] = __vsub4(grid[1] ^ s1, s1);
    }
    finish_two_halves<NT>(gv, b->scales[iqs / 2] & 0x0F, b->scales[iqs / 2] >> 4, __half2float(b->d), sb, xq, xs, nt, acc, tmask);
}

template <int NT> __device__ __forceinline__ void dot32m_iq3xxs(const uint8_t * row, int sb, const BlockQ8 * xq, int xs, int nt,
                                                               float * acc, int tmask) {
    const block_iq3_xxs * b = reinterpret_cast<const block_iq3_xxs *>(row) + sb / 8;
    const int iqs = 2 * (sb % 8);
    const int2 packed = make_int2(ld_b2(b->qs, iqs), ld_b2(b->qs, iqs + 1));
    const uint8_t * q3 = reinterpret_cast<const uint8_t *>(&packed);
    const uint32_t aux32 = ld_b2(b->qs, QK_K / 16 + iqs / 2);
    int gv[8];
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid = make_int2(iq3xxs_grid[q3[l0]], iq3xxs_grid[q3[l0 + 1]]);
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * l0 / 2));
        const int s0 = __vcmpne4(signs & 0x08040201, 0), s1 = __vcmpne4(signs & 0x80402010, 0);
        gv[l0] = __vsub4(grid.x ^ s0, s0);
        gv[l0 + 1] = __vsub4(grid.y ^ s1, s1);
    }
    const int ls = aux32 >> 28;
    const float d = __half2float(b->d);
    BL_TOK_LOOP {
        const BlockQ8 & q = xq[t * xs + sb];
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) sumi = __dp4a(gv[j], ld_b4(q.qs, j), sumi);
        sumi = (ls * sumi + sumi / 2) / 2;
        acc[t] += d * __low2float(q.ds) * sumi;
    }
}

template <int NT> __device__ __forceinline__ void dot32m_iq3s(const uint8_t * row, int sb, const BlockQ8 * xq, int xs, int nt,
                                                             float * acc, int tmask) {
    const block_iq3_s * b = reinterpret_cast<const block_iq3_s *>(row) + sb / 8;
    const int iqs = 2 * (sb % 8);
    const int2 packed = make_int2(ld_b2(b->qs, iqs), ld_b2(b->qs, iqs + 1));
    const uint8_t * qs = reinterpret_cast<const uint8_t *>(&packed);
    const int qh = b->qh[iqs / 2];
    const int sp32 = ld_b2(b->signs, iqs / 2);
    const uint8_t * sp8 = reinterpret_cast<const uint8_t *>(&sp32);
    int gv[8];
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid = make_int2(iq3s_grid[qs[l0] | ((qh << (8 - l0)) & 0x100)], iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
        const int s0 = __vcmpne4(((sp8[l0 / 2] & 0x03) << 7) | ((sp8[l0 / 2] & 0x0C) << 21), 0);
        const int s1 = __vcmpne4(((sp8[l0 / 2] & 0x30) << 3) | ((sp8[l0 / 2] & 0xC0) << 17), 0);
        gv[l0] = __vsub4(grid.x ^ s0, s0);
        gv[l0 + 1] = __vsub4(grid.y ^ s1, s1);
    }
    const int sc = 1 + 2 * ((b->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
    const float d = __half2float(b->d);
    BL_TOK_LOOP {
        const BlockQ8 & q = xq[t * xs + sb];
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) sumi = __dp4a(gv[j], ld_b4(q.qs, j), sumi);
        acc[t] += d * __low2float(q.ds) * (sumi * sc);
    }
}

template <int NT> __device__ __forceinline__ void dot32m_iq4nl(const uint8_t * row, int sb, const BlockQ8 * xq, int xs, int nt,
                                                              float * acc, int tmask) {
    const block_iq4_nl * b = reinterpret_cast<const block_iq4_nl *>(row) + sb;
    int2 v[4];
#pragma unroll
    for (int l = 0; l < 4; ++l) v[l] = table16(ld_b2(b->qs, l), reinterpret_cast<const int8_t *>(kvalues_iq4nl));
    const float d = __half2float(b->d);
    BL_TOK_LOOP {
        const BlockQ8 & q = xq[t * xs + sb];
        int sumi = 0;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            sumi = __dp4a(v[l].x, ld_b4(q.qs, l), sumi);
            sumi = __dp4a(v[l].y, ld_b4(q.qs, l + 4), sumi);
        }
        acc[t] += d * __low2float(q.ds) * sumi;
    }
}

template <int NT> __device__ __forceinline__ void dot32m_q20(const uint8_t * row, int sb, const BlockQ8 * xq, int xs, int nt,
                                                            float * acc, int tmask) {
    const block_q2_0 * b = reinterpret_cast<const block_q2_0 *>(row) + sb / 2;
    const int16_t * qs = reinterpret_cast<const int16_t *>(b->qs) + (sb % 2) * 4;
    int gv[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int v = qs[j];
        const int qe = __byte_perm(0x020100FF, 0x020100FF, v >> 0);
        const int qo = __byte_perm(0x020100FF, 0x020100FF, v >> 2);
        gv[2 * j] = static_cast<int>(__byte_perm(qe, qo, 0x5140));
        gv[2 * j + 1] = static_cast<int>(__byte_perm(qe, qo, 0x7362));
    }
    const float d = __half2float(b->d);
    BL_TOK_LOOP {
        const BlockQ8 & q = xq[t * xs + sb];
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) sumi = __dp4a(ld_b4(q.qs, j), gv[j], sumi);
        acc[t] += d * __low2float(q.ds) * sumi;
    }
}

// the dense formats (K-quants, IQ4_XS): sub-block sb of 32 values of a 256-value super-block
template <int NT> __device__ __forceinline__ void dot32m_q4k(const uint8_t * row, int sb, const BlockQ8 * xq, int xs, int nt,
                                                            float * acc, int tmask) {
    const block_q4_K * b = reinterpret_cast<const block_q4_K *>(row) + sb / 8;
    const int j = sb % 8;
    uint8_t sc, m;
    scale_min_k4(j, b->scales, sc, m);
    const uint8_t * qs = b->qs + 32 * (j / 2);
    const int sh = 4 * (j & 1);
    int gv[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) gv[i] = (ld_b4(qs, i) >> sh) & 0x0F0F0F0F;
    const float d = __half2float(b->data.d) * sc, dm = __half2float(b->data.dmin) * m;
    BL_TOK_LOOP {
        const BlockQ8 & q = xq[t * xs + sb];
        int sumi = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) sumi = __dp4a(gv[i], ld_b4(q.qs, i), sumi);
        acc[t] += d * __low2float(q.ds) * sumi - dm * __high2float(q.ds);
    }
}

template <int NT> __device__ __forceinline__ void dot32m_q5k(const uint8_t * row, int sb, const BlockQ8 * xq, int xs, int nt,
                                                            float * acc, int tmask) {
    const block_q5_K * b = reinterpret_cast<const block_q5_K *>(row) + sb / 8;
    const int j = sb % 8;
    uint8_t sc, m;
    scale_min_k4(j, b->scales, sc, m);
    const uint8_t * qs = b->qs + 32 * (j / 2);
    const int sh = 4 * (j & 1);
    int gv[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) gv[i] = ((ld_b4(qs, i) >> sh) & 0x0F0F0F0F) | (((ld_b4(b->qh, i) >> j) & 0x01010101) << 4);
    const float d = __half2float(b->data.d) * sc, dm = __half2float(b->data.dmin) * m;
    BL_TOK_LOOP {
        const BlockQ8 & q = xq[t * xs + sb];
        int sumi = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) sumi = __dp4a(gv[i], ld_b4(q.qs, i), sumi);
        acc[t] += d * __low2float(q.ds) * sumi - dm * __high2float(q.ds);
    }
}

template <int NT> __device__ __forceinline__ void dot32m_q6k(const uint8_t * row, int sb, const BlockQ8 * xq, int xs, int nt,
                                                            float * acc, int tmask) {
    const block_q6_K * b = reinterpret_cast<const block_q6_K *>(row) + sb / 8;   // 210-byte blocks: 16-bit loads
    const int h = (sb % 8) / 4, k = sb % 4;
    const uint8_t * ql = b->ql + 64 * h + 32 * (k & 1), * qh = b->qh + 32 * h;
    const int lsh = 4 * (k >> 1), hsh = 2 * k;
    int gv[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
        gv[i] = __vsubss4(((ld_b2(ql, i) >> lsh) & 0x0F0F0F0F) | (((ld_b2(qh, i) >> hsh) & 0x03030303) << 4), 0x20202020);
    const int8_t * scs = b->scales + 8 * h + 2 * k;
    const float d = __half2float(b->d);
    const int s0 = scs[0], s1 = scs[1];
    BL_TOK_LOOP {
        const BlockQ8 & q = xq[t * xs + sb];
        int a0 = 0, a1 = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            a0 = __dp4a(gv[i], ld_b4(q.qs, i), a0);
            a1 = __dp4a(gv[i + 4], ld_b4(q.qs, i + 4), a1);
        }
        acc[t] += d * __low2float(q.ds) * (s0 * a0 + s1 * a1);
    }
}

template <int NT> __device__ __forceinline__ void dot32m_iq4xs(const uint8_t * row, int sb, const BlockQ8 * xq, int xs, int nt,
                                                              float * acc, int tmask) {
    const block_iq4_xs * b = reinterpret_cast<const block_iq4_xs *>(row) + sb / 8;
    const int j = sb % 8;
    const int ls = ((b->scales_l[j / 2] >> 4 * (j % 2)) & 0xf) | (((b->scales_h >> 2 * j) & 3) << 4);
    int2 v[4];
#pragma unroll
    for (int l = 0; l < 4; ++l) v[l] = table16(ld_b4(b->qs + 16 * j, l), reinterpret_cast<const int8_t *>(kvalues_iq4nl));
    const float d = __half2float(b->d) * (ls - 32);
    BL_TOK_LOOP {
        const BlockQ8 & q = xq[t * xs + sb];
        int sumi = 0;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            sumi = __dp4a(v[l].x, ld_b4(q.qs, l), sumi);
            sumi = __dp4a(v[l].y, ld_b4(q.qs, l + 4), sumi);
        }
        acc[t] += d * __low2float(q.ds) * sumi;
    }
}
#undef BL_TOK_LOOP

template <int F, int NT> __device__ __forceinline__ void dot32m(const uint8_t * row, int sb, const BlockQ8 * xq, int xs, int nt,
                                                                float * acc, int tmask) {
    if constexpr (F == IQ2_XXS) dot32m_iq2xxs<NT>(row, sb, xq, xs, nt, acc, tmask);
    else if constexpr (F == IQ2_XS) dot32m_iq2xs<NT>(row, sb, xq, xs, nt, acc, tmask);
    else if constexpr (F == IQ2_S) dot32m_iq2s<NT>(row, sb, xq, xs, nt, acc, tmask);
    else if constexpr (F == IQ3_XXS) dot32m_iq3xxs<NT>(row, sb, xq, xs, nt, acc, tmask);
    else if constexpr (F == IQ3_S) dot32m_iq3s<NT>(row, sb, xq, xs, nt, acc, tmask);
    else if constexpr (F == IQ4_NL) dot32m_iq4nl<NT>(row, sb, xq, xs, nt, acc, tmask);
    else if constexpr (F == Q2_0) dot32m_q20<NT>(row, sb, xq, xs, nt, acc, tmask);
    else if constexpr (F == Q4_K) dot32m_q4k<NT>(row, sb, xq, xs, nt, acc, tmask);
    else if constexpr (F == Q5_K) dot32m_q5k<NT>(row, sb, xq, xs, nt, acc, tmask);
    else if constexpr (F == Q6_K) dot32m_q6k<NT>(row, sb, xq, xs, nt, acc, tmask);
    else if constexpr (F == IQ4_XS) dot32m_iq4xs<NT>(row, sb, xq, xs, nt, acc, tmask);
}

// formats with a dot32m
__host__ __device__ constexpr bool has_dp4a(uint32_t t) {
    return t == IQ2_XXS || t == IQ2_XS || t == IQ2_S || t == IQ3_XXS || t == IQ3_S || t == IQ4_NL || t == Q2_0 || t == Q4_K ||
           t == Q5_K || t == Q6_K || t == IQ4_XS;
}

}  // namespace bl::cuda
