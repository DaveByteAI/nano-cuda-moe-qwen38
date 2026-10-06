// AVX2 dot products of Q2_0 rows (ISTA-DASLab's 2-bit type: 64 values per block, fp16 d + 16 bytes, value = d*(q-1))
// against activations quantized to int8 per 32 values. ggml has no SIMD kernel for Q2_0 on x86 (0.7 GB/s/thread).
//
// Q2_0 stores value 4k+m of a block in bits 2m..2m+1 of byte k, so shifting the 16 bytes by 0/2/4/6 gives four
// "planes" of 16 values each with no shuffling. The activations are laid out in the same plane order when they are
// quantized (once per token), which makes the dot a plain maddubs.
#pragma once

#include <cstddef>
#include <cstdint>

namespace bl::cpu {

// activations of one row of K values (K a multiple of 64), quantized and plane-ordered
struct Q8Planes {
    struct Block {
        float  d[2];      // scale of values 0..31 and 32..63
        float  dsum[2];   // d * sum of the int8 values of each half (the "-1" of Q2_0)
        int8_t q[64];     // plane order: [m][k] = value 4k+m, k = 0..15
    };
};

size_t q8planes_bytes(int K);
void   q8planes_quantize(const float * x, int K, void * out);
// dot of one Q2_0 row (K values) with plane-ordered activations
float  q2_0_dot(const uint8_t * row, const void * q8p, int K);

}  // namespace bl::cpu
