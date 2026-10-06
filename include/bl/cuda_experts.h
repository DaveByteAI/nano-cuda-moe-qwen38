// Routed-expert kernels: one launch computes a batch of experts of one layer. An expert is stored contiguously as
// [gate | up | down] (gate/up: F rows of D values, down: D rows of F values, in the layer's formats).
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace bl::cuda {

constexpr int kMaxBatchExperts = 16;

struct ExpertBatch {
    const uint8_t * base[kMaxBatchExperts];   // device address of each expert's [gate|up|down]
    int             slot[kMaxBatchExperts];   // output row (selection order) each expert writes
    int             n = 0;
};

// act[slot][r] = silu(gate_r . x) * (up_r . x)   for r < F
void expert_gate_up(uint32_t type_gu, const ExpertBatch & b, size_t gu_bytes, int F, int D, const float * x,
                    float * act, cudaStream_t s);
// out[slot][r] = down_r . act[slot]   for r < D
void expert_down(uint32_t type_gu, uint32_t type_d, const ExpertBatch & b, size_t gu_bytes, int F, int D,
                 const float * act, float * out, cudaStream_t s);
// y = sum_i w[i] * out[i]  (i in selection order, so the sum does not depend on which experts were cached)
void expert_combine(const float * out, const float * w, int n, int D, float * y, cudaStream_t s);

}  // namespace bl::cuda
