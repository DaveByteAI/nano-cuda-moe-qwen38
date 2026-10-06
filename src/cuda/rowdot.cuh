// Row dot products against an activation vector staged in shared memory (decode GEMV building blocks).
#pragma once

#include "dequant.cuh"

namespace bl::cuda {

// stage x (K values) into shared memory interleaved: value 8g+j at xs[j*G + g] (G = K/8 groups). The 32 lanes of a
// warp take consecutive groups, so they read consecutive words (the plain layout is an 8-way bank conflict).
__device__ __forceinline__ void stage_interleaved(const float * __restrict__ x, float * xs, int K) {
    const int G = K / 8;
    for (int i = threadIdx.x; i < K; i += blockDim.x) xs[(i & 7) * G + (i >> 3)] = x[i];
}

// x sits in shared memory interleaved: value 8g+j at xs[j*G + g] (G = K/8 groups), so the 32 lanes of a warp,
// which take consecutive groups, read consecutive words (the plain layout is an 8-way bank conflict)
template <int T, int R>
__device__ __forceinline__ void rows_dot_smem(const uint8_t * const * rows, const float * xs, int K, int lane, float * acc) {
    const int G = K / 8;
    for (int gg = lane; gg < G; gg += 32) {
        const int blk = gg / Grp<T>::groups, g = gg % Grp<T>::groups;
        float v[R][8];
#pragma unroll
        for (int r = 0; r < R; ++r) dequant8<T>(rows[r] + static_cast<size_t>(blk) * Grp<T>::bytes, g, v[r]);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float xv = xs[j * G + gg];
#pragma unroll
            for (int r = 0; r < R; ++r) acc[r] += v[r][j] * xv;
        }
    }
}


}  // namespace bl::cuda
