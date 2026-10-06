// Sampling on the GPU, one block per logits row: the top_k candidates (at most kSampleMaxK), softmax at a temperature,
// the top_p nucleus, then - for speculative decoding with a deterministic (argmax) draft d - the rejection step that
// keeps the output distribution exactly the target's: d is accepted with probability p(d), else a token is drawn from
// p with d removed (renormalized). A row without a draft (draft < 0) is a plain draw from p.
#pragma once

#include <cuda_runtime.h>

namespace bl::cuda {

constexpr int kSampleMaxK = 256;

struct SampleParams {
    float temperature = 1.f;   // > 0
    float top_p = 1.f;         // (0, 1]
    int   top_k = 40;          // 1 .. kSampleMaxK (0: kSampleMaxK)
};

// logits [T][V]; draft[t] (device-readable) the draft to judge on row t or -1; uni [T][2] uniforms in [0, 1);
// out_tok[t] the token of row t (the draft when accepted), out_acc[t] 1 when the draft was accepted
void sample_rows(const float * logits, int V, int T, const SampleParams & p, const int * draft, const float * uni, int * out_tok,
                 int * out_acc, cudaStream_t s);

}  // namespace bl::cuda
