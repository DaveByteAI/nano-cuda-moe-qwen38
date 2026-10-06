// A 32-value sub-block of an expert row as int8 values plus the float scale of each 16-value half:
//   value[k] = int8(gv bytes)[k] * (k < 16 ? s0 : s1)
// The same decoding as dot32m (dp4a.cuh), with ggml's integer scale formulas turned into exact float factors
// (ggml's dot products truncate the scaled integer sums; the dequantized values are these).
#pragma once

#include "dp4a.cuh"

namespace bl::cuda {

template <int F> __device__ __forceinline__ void decode32(const uint8_t * row, int sb, int * gv, float & s0, float & s1);

template <> __device__ __forceinline__ void decode32<IQ2_XXS>(const uint8_t * row, int sb, int * gv, float & s0, float & s1) {
    const block_iq2_xxs * b = reinterpret_cast<const block_iq2_xxs *>(row) + sb / 8;
    const int iqs = 2 * (sb % 8);
    const int q2 = ld_b2(b->qs, iqs);
    const uint8_t * aux8 = reinterpret_cast<const uint8_t *>(&q2);
    const uint32_t aux32 = ld_b2(b->qs, iqs + 1);
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const uint2 grid = reinterpret_cast<const uint2 *>(iq2xxs_grid)[aux8[k0 / 2]];
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));
        const int a = __vcmpne4(signs & 0x08040201, 0), c = __vcmpne4(signs & 0x80402010, 0);
        gv[k0] = __vsub4(grid.x ^ a, a);
        gv[k0 + 1] = __vsub4(grid.y ^ c, c);
    }
    s0 = s1 = __half2float(b->d) * static_cast<float>(aux32 >> 27 | 1) * 0.125f;
}

template <> __device__ __forceinline__ void decode32<IQ2_XS>(const uint8_t * row, int sb, int * gv, float & s0, float & s1) {
    const block_iq2_xs * b = reinterpret_cast<const block_iq2_xs *>(row) + sb / 8;
    const int iqs = 2 * (sb % 8);
    const int2 packed = make_int2(ld_b2(b->qs, iqs), ld_b2(b->qs, iqs + 1));
    const uint16_t * q2 = reinterpret_cast<const uint16_t *>(&packed);
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint2 grid = reinterpret_cast<const uint2 *>(iq2xs_grid)[q2[l0 / 2] & 0x1FF];
        const uint32_t signs = unpack_ksigns(q2[l0 / 2] >> 9);
        const int a = __vcmpne4(signs & 0x08040201, 0), c = __vcmpne4(signs & 0x80402010, 0);
        gv[l0] = __vsub4(grid.x ^ a, a);
        gv[l0 + 1] = __vsub4(grid.y ^ c, c);
    }
    const float d = __half2float(b->d) * 0.125f;
    s0 = d * static_cast<float>(2 * (b->scales[iqs / 2] & 0x0F) + 1);
    s1 = d * static_cast<float>(2 * (b->scales[iqs / 2] >> 4) + 1);
}

template <> __device__ __forceinline__ void decode32<IQ2_S>(const uint8_t * row, int sb, int * gv, float & s0, float & s1) {
    const block_iq2_s * b = reinterpret_cast<const block_iq2_s *>(row) + sb / 8;
    const int iqs = 2 * (sb % 8);
    const int qs_packed = ld_b2(b->qs, iqs / 2);
    const uint8_t * qs = reinterpret_cast<const uint8_t *>(&qs_packed);
    const int qh = b->qh[iqs / 2];
    const int sp32 = ld_b2(b->qs, QK_K / 32 + iqs / 2);
    const uint8_t * sp8 = reinterpret_cast<const uint8_t *>(&sp32);
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int * grid = reinterpret_cast<const int *>(iq2s_grid + (qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)));
        const int a = __vcmpne4(((sp8[l0 / 2] & 0x03) << 7) | ((sp8[l0 / 2] & 0x0C) << 21), 0);
        const int c = __vcmpne4(((sp8[l0 / 2] & 0x30) << 3) | ((sp8[l0 / 2] & 0xC0) << 17), 0);
        gv[l0] = __vsub4(grid[0] ^ a, a);
        gv[l0 + 1] = __vsub4(grid[1] ^ c, c);
    }
    const float d = __half2float(b->d) * 0.125f;
    s0 = d * static_cast<float>(2 * (b->scales[iqs / 2] & 0x0F) + 1);
    s1 = d * static_cast<float>(2 * (b->scales[iqs / 2] >> 4) + 1);
}

template <> __device__ __forceinline__ void decode32<IQ3_XXS>(const uint8_t * row, int sb, int * gv, float & s0, float & s1) {
    const block_iq3_xxs * b = reinterpret_cast<const block_iq3_xxs *>(row) + sb / 8;
    const int iqs = 2 * (sb % 8);
    const int2 packed = make_int2(ld_b2(b->qs, iqs), ld_b2(b->qs, iqs + 1));
    const uint8_t * q3 = reinterpret_cast<const uint8_t *>(&packed);
    const uint32_t aux32 = ld_b2(b->qs, QK_K / 16 + iqs / 2);
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid = make_int2(iq3xxs_grid[q3[l0]], iq3xxs_grid[q3[l0 + 1]]);
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * l0 / 2));
        const int a = __vcmpne4(signs & 0x08040201, 0), c = __vcmpne4(signs & 0x80402010, 0);
        gv[l0] = __vsub4(grid.x ^ a, a);
        gv[l0 + 1] = __vsub4(grid.y ^ c, c);
    }
    s0 = s1 = __half2float(b->d) * static_cast<float>(2 * (aux32 >> 28) + 1) * 0.25f;
}

template <> __device__ __forceinline__ void decode32<IQ3_S>(const uint8_t * row, int sb, int * gv, float & s0, float & s1) {
    const block_iq3_s * b = reinterpret_cast<const block_iq3_s *>(row) + sb / 8;
    const int iqs = 2 * (sb % 8);
    const int2 packed = make_int2(ld_b2(b->qs, iqs), ld_b2(b->qs, iqs + 1));
    const uint8_t * qs = reinterpret_cast<const uint8_t *>(&packed);
    const int qh = b->qh[iqs / 2];
    const int sp32 = ld_b2(b->signs, iqs / 2);
    const uint8_t * sp8 = reinterpret_cast<const uint8_t *>(&sp32);
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid = make_int2(iq3s_grid[qs[l0] | ((qh << (8 - l0)) & 0x100)], iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
        const int a = __vcmpne4(((sp8[l0 / 2] & 0x03) << 7) | ((sp8[l0 / 2] & 0x0C) << 21), 0);
        const int c = __vcmpne4(((sp8[l0 / 2] & 0x30) << 3) | ((sp8[l0 / 2] & 0xC0) << 17), 0);
        gv[l0] = __vsub4(grid.x ^ a, a);
        gv[l0 + 1] = __vsub4(grid.y ^ c, c);
    }
    s0 = s1 = __half2float(b->d) * static_cast<float>(1 + 2 * ((b->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F));
}

template <> __device__ __forceinline__ void decode32<IQ4_NL>(const uint8_t * row, int sb, int * gv, float & s0, float & s1) {
    const block_iq4_nl * b = reinterpret_cast<const block_iq4_nl *>(row) + sb;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const int2 v = table16(ld_b2(b->qs, l), reinterpret_cast<const int8_t *>(kvalues_iq4nl));
        gv[l] = v.x;       // values 4l .. 4l+3
        gv[l + 4] = v.y;   // values 16+4l ..
    }
    s0 = s1 = __half2float(b->d);
}

// sub-block sb of a 256-value super-block: IQ4_NL's 16 bytes of codes, each sub-block its own 6-bit scale
template <> __device__ __forceinline__ void decode32<IQ4_XS>(const uint8_t * row, int sb, int * gv, float & s0, float & s1) {
    const block_iq4_xs * b = reinterpret_cast<const block_iq4_xs *>(row) + sb / 8;
    const int ib = sb % 8;
    const uint8_t * qs = b->qs + 16 * ib;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const int2 v = table16(ld_b2(qs, l), reinterpret_cast<const int8_t *>(kvalues_iq4nl));
        gv[l] = v.x;       // values 4l .. 4l+3
        gv[l + 4] = v.y;   // values 16+4l ..
    }
    const int ls = ((b->scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((b->scales_h >> 2 * ib) & 3) << 4);
    s0 = s1 = __half2float(b->d) * static_cast<float>(ls - 32);
}

template <> __device__ __forceinline__ void decode32<Q2_0>(const uint8_t * row, int sb, int * gv, float & s0, float & s1) {
    const block_q2_0 * b = reinterpret_cast<const block_q2_0 *>(row) + sb / 2;
    const int16_t * qs = reinterpret_cast<const int16_t *>(b->qs) + (sb % 2) * 4;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int v = qs[j];
        const int qe = __byte_perm(0x020100FF, 0x020100FF, v >> 0);
        const int qo = __byte_perm(0x020100FF, 0x020100FF, v >> 2);
        gv[2 * j] = static_cast<int>(__byte_perm(qe, qo, 0x5140));
        gv[2 * j + 1] = static_cast<int>(__byte_perm(qe, qo, 0x7362));
    }
    s0 = s1 = __half2float(b->d);
}

}  // namespace bl::cuda
