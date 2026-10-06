// AVX2 dot products of the i-quant expert rows (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S) with several tokens at once.
// ggml-cpu's own AVX2 vec_dot is one token per call: each token re-does the codebook lookups and sign expansion of
// every 32-value chunk. Here a chunk is decoded once (grid magnitudes, a +-1 sign vector, the integer scale) and every
// token applies it with a sign, a maddubs, a madd and an add. Activations: ggml's block_q8_K rows (what ggml's
// vec_dot for these formats takes). The integer arithmetic is ggml's; only the order of the float sums differs.
#pragma once

#include <cstddef>
#include <cstdint>

namespace bl::cpu {

constexpr int kIqMaxTokens = 8;

bool iq_supported(uint32_t ggml_type);

// g[t] = gate_row . x[t], u[t] = up_row . x[t] for nt tokens (nt <= kIqMaxTokens); K values per row, x[t] q8_K rows
void iq_gate_up(uint32_t ggml_type, const uint8_t * gate_row, const uint8_t * up_row, int K, const void * const * x, int nt,
                float * g, float * u);

}  // namespace bl::cpu
