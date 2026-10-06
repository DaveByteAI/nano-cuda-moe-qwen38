// Reading and writing the QSA KV cache (bl::cuda::KvCache): fp16 values, or int8 with one fp16 scale per 32 values.
// Rows are [cell][kv head][256]; a warp lane handles 8 consecutive values (dims 8l .. 8l+7, scale group l / 4).
#pragma once

#include <cuda_fp16.h>

#include "bl/cuda_window.h"

namespace bl::cuda {

constexpr int kKvD = 256;

__device__ __forceinline__ size_t kv_row(const KvCache & c, int pos, int kv_heads, int h) {
    return static_cast<size_t>(pos & c.mask) * kv_heads + h;
}

// the 8 values of lane `lane` in row `row` of k (which = 0) or v (which = 1)
template <bool Q8> __device__ __forceinline__ void kv_ld8(const KvCache & c, int which, size_t row, int lane, float * o) {
    if constexpr (Q8) {
        const int8_t * base = static_cast<const int8_t *>(which ? c.v : c.k);
        const uint2 u = *reinterpret_cast<const uint2 *>(base + row * kKvD + 8 * lane);
        const float s = __half2float((which ? c.sv : c.sk)[row * (kKvD / kKvGroup) + (lane >> 2)]);
        const int8_t * b = reinterpret_cast<const int8_t *>(&u);
#pragma unroll
        for (int i = 0; i < 8; ++i) o[i] = s * static_cast<float>(b[i]);
    } else {
        const __half * base = static_cast<const __half *>(which ? c.v : c.k);
        const uint4 u = *reinterpret_cast<const uint4 *>(base + row * kKvD + 8 * lane);
        const __half2 * h = reinterpret_cast<const __half2 *>(&u);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 f = __half22float2(h[i]);
            o[2 * i] = f.x;
            o[2 * i + 1] = f.y;
        }
    }
}

// value x of dim d (one thread per dim, blockDim = 256, so a warp is one scale group) into row `row`
template <bool Q8> __device__ __forceinline__ void kv_st(const KvCache & c, int which, size_t row, int d, float x) {
    if constexpr (Q8) {
        float a = fabsf(x);
        for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
        const __half sh = __float2half_rn(a / 127.f);
        const float s = __half2float(sh);
        const float qv = s > 0.f ? fminf(fmaxf(rintf(x / s), -127.f), 127.f) : 0.f;
        static_cast<int8_t *>(which ? c.v : c.k)[row * kKvD + d] = static_cast<int8_t>(qv);
        if ((d & 31) == 0) (which ? c.sv : c.sk)[row * (kKvD / kKvGroup) + d / kKvGroup] = sh;
    } else {
        static_cast<__half *>(which ? c.v : c.k)[row * kKvD + d] = __float2half_rn(x);
    }
}

}  // namespace bl::cuda
