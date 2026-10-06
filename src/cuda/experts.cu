// Batched routed-expert kernels (phase 2): blockIdx.y picks the expert, one warp per output row.
#include <stdexcept>
#include <string>

#include "bl/cuda_experts.h"
#include "dequant.cuh"

namespace bl::cuda {

namespace {

constexpr int kRows = 4;   // warps (rows) per block

template <int T>
__device__ __forceinline__ float row_dot(const uint8_t * row, const float * x, int K) {
    const int lane = threadIdx.x & 31;
    float acc = 0.f;
    for (int gg = lane; gg < K / 8; gg += 32) {
        const int blk = gg / Grp<T>::groups, g = gg % Grp<T>::groups;
        float v[8];
        dequant8<T>(row + static_cast<size_t>(blk) * Grp<T>::bytes, g, v);
        const float4 a = reinterpret_cast<const float4 *>(x)[2 * gg];
        const float4 b = reinterpret_cast<const float4 *>(x)[2 * gg + 1];
        acc += v[0] * a.x + v[1] * a.y + v[2] * a.z + v[3] * a.w + v[4] * b.x + v[5] * b.y + v[6] * b.z + v[7] * b.w;
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    return acc;
}

template <int T>
__global__ void gate_up_k(ExpertBatch b, size_t gu_bytes, int F, int D, const float * __restrict__ x, float * act) {
    const int e = blockIdx.y, r = blockIdx.x * kRows + (threadIdx.x >> 5);
    if (r >= F) return;
    const size_t rb = gu_bytes / F;
    const uint8_t * gate = b.base[e] + static_cast<size_t>(r) * rb;
    const float g = row_dot<T>(gate, x, D);
    const float u = row_dot<T>(gate + gu_bytes, x, D);
    if ((threadIdx.x & 31) == 0) act[static_cast<size_t>(b.slot[e]) * F + r] = g / (1.f + expf(-g)) * u;
}

template <int T>
__global__ void down_k(ExpertBatch b, size_t gu_bytes, size_t d_rb, int F, int D, const float * __restrict__ act, float * out) {
    const int e = blockIdx.y, r = blockIdx.x * kRows + (threadIdx.x >> 5);
    if (r >= D) return;
    const uint8_t * row = b.base[e] + 2 * gu_bytes + static_cast<size_t>(r) * d_rb;
    const float v = row_dot<T>(row, act + static_cast<size_t>(b.slot[e]) * F, F);
    if ((threadIdx.x & 31) == 0) out[static_cast<size_t>(b.slot[e]) * D + r] = v;
}

__global__ void combine_k(const float * out, const float * w, int n, int D, float * y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= D) return;
    float acc = 0.f;
    for (int e = 0; e < n; ++e) acc += w[e] * out[static_cast<size_t>(e) * D + i];
    y[i] = acc;
}

size_t row_bytes_of(uint32_t t, int K) {
    switch (t) {
        case Q2_0:    return K / 64 * 18;
        case IQ4_NL:  return K / 32 * 18;
        case IQ2_XXS: return K / 256 * 66;
        case IQ2_XS:  return K / 256 * 74;
        case IQ2_S:   return K / 256 * 82;
        case IQ3_XXS: return K / 256 * 98;
        case IQ3_S:   return K / 256 * 110;
        default: throw std::runtime_error("expert format " + std::to_string(t) + " not supported");
    }
}

template <template <int> class F, typename... A>
void dispatch_expert(uint32_t t, A... a) {
    switch (t) {
        case IQ2_XXS: return F<IQ2_XXS>::run(a...);
        case IQ2_XS:  return F<IQ2_XS>::run(a...);
        case IQ2_S:   return F<IQ2_S>::run(a...);
        case IQ3_XXS: return F<IQ3_XXS>::run(a...);
        case IQ3_S:   return F<IQ3_S>::run(a...);
        case IQ4_NL:  return F<IQ4_NL>::run(a...);
        case Q2_0:    return F<Q2_0>::run(a...);
        default: throw std::runtime_error("expert format " + std::to_string(t) + " not supported");
    }
}

template <int T> struct GateUp {
    static void run(const ExpertBatch & b, size_t gu, int F, int D, const float * x, float * act, cudaStream_t s) {
        gate_up_k<T><<<dim3((F + kRows - 1) / kRows, b.n), 32 * kRows, 0, s>>>(b, gu, F, D, x, act);
    }
};
template <int T> struct Down {
    static void run(const ExpertBatch & b, size_t gu, size_t d_rb, int F, int D, const float * act, float * out, cudaStream_t s) {
        down_k<T><<<dim3((D + kRows - 1) / kRows, b.n), 32 * kRows, 0, s>>>(b, gu, d_rb, F, D, act, out);
    }
};

}  // namespace

void expert_gate_up(uint32_t t, const ExpertBatch & b, size_t gu, int F, int D, const float * x, float * act, cudaStream_t s) {
    if (b.n == 0) return;
    dispatch_expert<GateUp>(t, b, gu, F, D, x, act, s);
}

void expert_down(uint32_t, uint32_t td, const ExpertBatch & b, size_t gu, int F, int D, const float * act, float * out,
                 cudaStream_t s) {
    if (b.n == 0) return;
    dispatch_expert<Down>(td, b, gu, row_bytes_of(td, F), F, D, act, out, s);
}

void expert_combine(const float * out, const float * w, int n, int D, float * y, cudaStream_t s) {
    combine_k<<<(D + 255) / 256, 256, 0, s>>>(out, w, n, D, y);
}

}  // namespace bl::cuda
