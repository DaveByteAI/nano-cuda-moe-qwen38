// Device-side dequantization of every weight format in the Qwen3.8-Flash-Next GSQ-RCO files.
//
// dequant8<T>(block, g, y) writes the 8 consecutive values of group g of one block (a block of B values has B/8
// groups). Each is a transcription of ggml's dequantize_row_* (ggml-quants.c at llama.cpp 3cf0325) restricted to one
// group, with the same float operations in the same order, so the values match ggml's to_float bit for bit. The
// K-quants' `scale*q - min` is written with __fmul_rn/__fsub_rn so nvcc cannot contract it into an FMA.
#pragma once

#include <cstdint>

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"

namespace bl::cuda {

enum Fmt : int {
    F32 = 0, F16 = 1, Q8_0 = 8, Q4_K = 12, Q5_K = 13, Q6_K = 14, IQ2_XXS = 16, IQ2_XS = 17, IQ3_XXS = 18,
    IQ4_NL = 20, IQ3_S = 21, IQ2_S = 22, IQ4_XS = 23, BF16 = 30, Q2_0 = 42,
};

template <int T> struct Geo;
template <> struct Geo<F32>     { static constexpr int elems = 1,     bytes = 4; };
template <> struct Geo<F16>     { static constexpr int elems = 1,     bytes = 2; };
template <> struct Geo<BF16>    { static constexpr int elems = 1,     bytes = 2; };
template <> struct Geo<Q8_0>    { static constexpr int elems = QK8_0, bytes = sizeof(block_q8_0); };
template <> struct Geo<Q4_K>    { static constexpr int elems = QK_K,  bytes = sizeof(block_q4_K); };
template <> struct Geo<Q5_K>    { static constexpr int elems = QK_K,  bytes = sizeof(block_q5_K); };
template <> struct Geo<Q6_K>    { static constexpr int elems = QK_K,  bytes = sizeof(block_q6_K); };
template <> struct Geo<IQ2_XXS> { static constexpr int elems = QK_K,  bytes = sizeof(block_iq2_xxs); };
template <> struct Geo<IQ2_XS>  { static constexpr int elems = QK_K,  bytes = sizeof(block_iq2_xs); };
template <> struct Geo<IQ2_S>   { static constexpr int elems = QK_K,  bytes = sizeof(block_iq2_s); };
template <> struct Geo<IQ3_XXS> { static constexpr int elems = QK_K,  bytes = sizeof(block_iq3_xxs); };
template <> struct Geo<IQ3_S>   { static constexpr int elems = QK_K,  bytes = sizeof(block_iq3_s); };
template <> struct Geo<IQ4_NL>  { static constexpr int elems = QK4_NL, bytes = sizeof(block_iq4_nl); };
template <> struct Geo<IQ4_XS>  { static constexpr int elems = QK_K,  bytes = sizeof(block_iq4_xs); };
template <> struct Geo<Q2_0>    { static constexpr int elems = QK2_0, bytes = sizeof(block_q2_0); };

// "groups of 8": for the plain float types a "block" is taken as 8 values so every format has the same shape
template <int T> struct Grp {
    static constexpr int elems = Geo<T>::elems == 1 ? 8 : Geo<T>::elems;             // values per addressable unit
    static constexpr int bytes = Geo<T>::elems == 1 ? 8 * Geo<T>::bytes : Geo<T>::bytes;
    static constexpr int groups = elems / 8;
};

__device__ __forceinline__ float h2f(half h) { return __half2float(h); }

// 8 bytes from a 2-byte aligned address as four 16-bit loads (most block sizes are only 2-byte aligned)
__device__ __forceinline__ uint64_t ld8_a2(const uint8_t * p) {
    const uint16_t * q = reinterpret_cast<const uint16_t *>(p);
    return static_cast<uint64_t>(q[0]) | static_cast<uint64_t>(q[1]) << 16 | static_cast<uint64_t>(q[2]) << 32 |
           static_cast<uint64_t>(q[3]) << 48;
}
__device__ __forceinline__ uint64_t ld8_a8(const uint8_t * p) { return *reinterpret_cast<const uint64_t *>(p); }
__device__ __forceinline__ uint32_t byte_of(uint64_t v, int j) { return static_cast<uint32_t>(v >> (8 * j)) & 0xff; }
// kmask_iq2xs[j] == 1 << j: a bit test, not a load from a device table
__device__ __forceinline__ float sgn(uint32_t signs, int j) { return ((signs >> j) & 1) ? -1.f : 1.f; }

__device__ __forceinline__ void scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

template <int T> __device__ __forceinline__ void dequant8(const uint8_t * blk, int g, float * y);

template <> __device__ __forceinline__ void dequant8<F32>(const uint8_t * blk, int, float * y) {
    const float4 a = reinterpret_cast<const float4 *>(blk)[0], b = reinterpret_cast<const float4 *>(blk)[1];
    y[0] = a.x; y[1] = a.y; y[2] = a.z; y[3] = a.w; y[4] = b.x; y[5] = b.y; y[6] = b.z; y[7] = b.w;
}
template <> __device__ __forceinline__ void dequant8<F16>(const uint8_t * blk, int, float * y) {
    const half * p = reinterpret_cast<const half *>(blk);
    for (int j = 0; j < 8; ++j) y[j] = __half2float(p[j]);
}
template <> __device__ __forceinline__ void dequant8<BF16>(const uint8_t * blk, int, float * y) {
    const uint4 q = *reinterpret_cast<const uint4 *>(blk);   // a row of bf16 is 16-byte aligned in 8-value groups
    const uint32_t w[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        y[2 * i]     = __uint_as_float(w[i] << 16);
        y[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
    }
}

template <> __device__ __forceinline__ void dequant8<Q8_0>(const uint8_t * blk, int g, float * y) {
    const block_q8_0 * x = reinterpret_cast<const block_q8_0 *>(blk);
    const float d = h2f(x->d);
    for (int j = 0; j < 8; ++j) y[j] = x->qs[8 * g + j] * d;
}

template <> __device__ __forceinline__ void dequant8<Q2_0>(const uint8_t * blk, int g, float * y) {
    const block_q2_0 * x = reinterpret_cast<const block_q2_0 *>(blk);
    const float d = h2f(x->d);
    for (int j = 0; j < 8; ++j) {
        const int e = 8 * g + j;
        const int q = (x->qs[e / 4] >> ((e % 4) * 2)) & 0x03;
        y[j] = (q - 1) * d;
    }
}

template <> __device__ __forceinline__ void dequant8<IQ4_NL>(const uint8_t * blk, int g, float * y) {
    const block_iq4_nl * x = reinterpret_cast<const block_iq4_nl *>(blk);
    const float d = h2f(x->d);
    const uint8_t * qs = x->qs + 8 * (g & 1);
    for (int j = 0; j < 8; ++j) y[j] = d * kvalues_iq4nl[g < 2 ? (qs[j] & 0xf) : (qs[j] >> 4)];
}

template <> __device__ __forceinline__ void dequant8<IQ4_XS>(const uint8_t * blk, int g, float * y) {
    const block_iq4_xs * x = reinterpret_cast<const block_iq4_xs *>(blk);
    const int ib = g / 4, sub = g % 4;
    const int ls = ((x->scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((x->scales_h >> 2 * ib) & 3) << 4);
    const float dl = h2f(x->d) * (ls - 32);
    const uint64_t q = ld8_a8(x->qs + 16 * ib + 8 * (sub & 1));   // blocks are 136 bytes: qs is 8-byte aligned
    const int sh = sub < 2 ? 0 : 4;
#pragma unroll
    for (int j = 0; j < 8; ++j) y[j] = dl * kvalues_iq4nl[(byte_of(q, j) >> sh) & 0xf];
}

template <> __device__ __forceinline__ void dequant8<Q4_K>(const uint8_t * blk, int g, float * y) {
    const block_q4_K * x = reinterpret_cast<const block_q4_K *>(blk);
    const int c64 = g / 8, hi = (g / 4) & 1, off = (g % 4) * 8;
    uint8_t sc, m;
    scale_min_k4(2 * c64 + hi, x->scales, sc, m);
    const float d1 = __fmul_rn(h2f(x->data.d), sc), m1 = __fmul_rn(h2f(x->data.dmin), m);
    const uint64_t q = ld8_a8(x->qs + 32 * c64 + off);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint32_t b = byte_of(q, j);
        y[j] = __fsub_rn(__fmul_rn(d1, hi ? (b >> 4) : (b & 0xF)), m1);
    }
}

template <> __device__ __forceinline__ void dequant8<Q5_K>(const uint8_t * blk, int g, float * y) {
    const block_q5_K * x = reinterpret_cast<const block_q5_K *>(blk);
    const int c64 = g / 8, hi = (g / 4) & 1, off = (g % 4) * 8;
    uint8_t sc, m;
    scale_min_k4(2 * c64 + hi, x->scales, sc, m);
    const float d1 = __fmul_rn(h2f(x->data.d), sc), m1 = __fmul_rn(h2f(x->data.dmin), m);
    const uint32_t u = 1u << (2 * c64 + hi);
    const uint64_t ql = ld8_a8(x->qs + 32 * c64 + off), qh = ld8_a8(x->qh + off);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint32_t b = byte_of(ql, j);
        const int q = static_cast<int>(hi ? (b >> 4) : (b & 0xF)) + ((byte_of(qh, j) & u) ? 16 : 0);
        y[j] = __fsub_rn(__fmul_rn(d1, q), m1);
    }
}

template <> __device__ __forceinline__ void dequant8<Q6_K>(const uint8_t * blk, int g, float * y) {
    const block_q6_K * x = reinterpret_cast<const block_q6_K *>(blk);
    const int n = g / 16, quarter = (g % 16) / 4, l0 = (g % 4) * 8;
    const float d = h2f(x->d);
    // the 8 values share one scale (l0..l0+7 never cross a 16-boundary); 210-byte blocks: 16-bit loads
    const float ds = d * x->scales[8 * n + l0 / 16 + 2 * quarter];
    const uint64_t ql = ld8_a2(x->ql + 64 * n + (quarter & 1) * 32 + l0);
    const uint64_t qh = ld8_a2(x->qh + 32 * n + l0);
    const int lo_shift = (quarter >> 1) * 4, hi_shift = 2 * quarter;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int q = static_cast<int>((byte_of(ql, j) >> lo_shift) & 0xF) | static_cast<int>(((byte_of(qh, j) >> hi_shift) & 3) << 4);
        y[j] = ds * static_cast<int8_t>(q - 32);
    }
}

template <> __device__ __forceinline__ void dequant8<IQ2_XXS>(const uint8_t * blk, int g, float * y) {
    const block_iq2_xxs * x = reinterpret_cast<const block_iq2_xxs *>(blk);
    const int ib32 = g / 4, l = g % 4;
    const uint16_t * q2 = x->qs + 4 * ib32;
    const uint32_t aux0 = q2[0] | (static_cast<uint32_t>(q2[1]) << 16);
    const uint32_t aux1 = q2[2] | (static_cast<uint32_t>(q2[3]) << 16);
    const float db = h2f(x->d) * (0.5f + (aux1 >> 28)) * 0.25f;
    const uint8_t * grid = reinterpret_cast<const uint8_t *>(iq2xxs_grid + ((aux0 >> 8 * l) & 0xff));
    const uint8_t signs = ksigns_iq2xs[(aux1 >> 7 * l) & 127];
    for (int j = 0; j < 8; ++j) y[j] = db * grid[j] * sgn(signs, j);
}

template <> __device__ __forceinline__ void dequant8<IQ2_XS>(const uint8_t * blk, int g, float * y) {
    const block_iq2_xs * x = reinterpret_cast<const block_iq2_xs *>(blk);
    const int ib32 = g / 4, l = g % 4;
    const float db = h2f(x->d) * (0.5f + ((x->scales[ib32] >> 4 * (l / 2)) & 0xf)) * 0.25f;
    const uint16_t q = x->qs[4 * ib32 + l];
    const uint8_t * grid = reinterpret_cast<const uint8_t *>(iq2xs_grid + (q & 511));
    const uint8_t signs = ksigns_iq2xs[q >> 9];
    for (int j = 0; j < 8; ++j) y[j] = db * grid[j] * sgn(signs, j);
}

template <> __device__ __forceinline__ void dequant8<IQ2_S>(const uint8_t * blk, int g, float * y) {
    const block_iq2_s * x = reinterpret_cast<const block_iq2_s *>(blk);
    const int ib32 = g / 4, l = g % 4;
    const float dl = h2f(x->d) * (0.5f + ((x->scales[ib32] >> 4 * (l / 2)) & 0xf)) * 0.25f;
    const int idx = x->qs[4 * ib32 + l] | ((x->qh[ib32] << (8 - 2 * l)) & 0x300);
    const uint8_t * grid = reinterpret_cast<const uint8_t *>(iq2s_grid + idx);
    const uint8_t signs = x->qs[QK_K / 8 + 4 * ib32 + l];
    for (int j = 0; j < 8; ++j) y[j] = dl * grid[j] * sgn(signs, j);
}

template <> __device__ __forceinline__ void dequant8<IQ3_XXS>(const uint8_t * blk, int g, float * y) {
    const block_iq3_xxs * x = reinterpret_cast<const block_iq3_xxs *>(blk);
    const int ib32 = g / 4, l = g % 4;
    const uint8_t * qs = x->qs + 8 * ib32;
    const uint8_t * ss = x->qs + QK_K / 4 + 4 * ib32;
    const uint32_t aux32 = ss[0] | (ss[1] << 8) | (ss[2] << 16) | (static_cast<uint32_t>(ss[3]) << 24);
    const float db = h2f(x->d) * (0.5f + (aux32 >> 28)) * 0.5f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * l) & 127];
    const uint8_t * grid1 = reinterpret_cast<const uint8_t *>(iq3xxs_grid + qs[2 * l + 0]);
    const uint8_t * grid2 = reinterpret_cast<const uint8_t *>(iq3xxs_grid + qs[2 * l + 1]);
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = db * grid1[j] * sgn(signs, j + 0);
        y[j + 4] = db * grid2[j] * sgn(signs, j + 4);
    }
}

template <> __device__ __forceinline__ void dequant8<IQ3_S>(const uint8_t * blk, int g, float * y) {
    const block_iq3_s * x = reinterpret_cast<const block_iq3_s *>(blk);
    const int ib32 = g / 4, l = g % 4;
    const float db = h2f(x->d) * (1 + 2 * ((x->scales[ib32 / 2] >> 4 * (ib32 % 2)) & 0xf));
    const uint8_t * qs = x->qs + 8 * ib32;
    const uint8_t qh = x->qh[ib32];
    const uint8_t signs = x->signs[4 * ib32 + l];
    const uint8_t * grid1 = reinterpret_cast<const uint8_t *>(iq3s_grid + (qs[2 * l + 0] | ((qh << (8 - 2 * l)) & 256)));
    const uint8_t * grid2 = reinterpret_cast<const uint8_t *>(iq3s_grid + (qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 256)));
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = db * grid1[j] * sgn(signs, j + 0);
        y[j + 4] = db * grid2[j] * sgn(signs, j + 4);
    }
}

}  // namespace bl::cuda
