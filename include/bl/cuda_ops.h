// Element-wise and small fused kernels of the qwen4exp forward (one token, fp32). Formulas: docs/MODEL.md; each
// kernel names the llama.cpp qwen4exp.cpp / ggml code it reproduces.
#pragma once

#include <cuda_runtime.h>

namespace bl::cuda {

// y[r] = x[r] / sqrt(mean(x[r]^2) + eps) * w[r % w_rows] * post   (rows of D; w: nullptr, or w_rows rows of D:
// w_rows <= 1 shares one row)
void rmsnorm_rows(const float * x, float * y, int rows, int D, const float * w, int w_rows, float eps, float post,
                  cudaStream_t s);

void silu_scale(float * x, int n, float scale, cudaStream_t s);                     // x = silu(x * scale)
void sigmoid_mul(float * x, const float * gate, int n, cudaStream_t s);             // x *= sigmoid(gate)
void sigmoid_scale(float * x, const float * scalar, int n, cudaStream_t s);         // x *= sigmoid(*scalar)
void swiglu(const float * g, const float * u, float * out, int n, cudaStream_t s);   // out = silu(g) * u
void axpy(float * y, const float * x, float a, int n, cudaStream_t s);              // y += a * x
void add(float * y, const float * x, int n, cudaStream_t s);                        // y += x
void repeat(const float * x, float * y, int D, int times, cudaStream_t s);          // y[c*D + i] = x[i]

// hyper-connections (qwen4exp.cpp build_hc_mix / build_hc_combine)
void hc_gate_mean(const float * xn, const float * gate, float * mixed, int D, int hc, cudaStream_t s);
void hc_combine(float * R, const float * h, const float * inj, int D, int hc, cudaStream_t s);

// GDN (build_layer_attn_linear + ggml gated_delta_net)
// conv over [3 history | x] per channel, silu; the history shifts in place. state: [C][3], w: [C][4]
void gdn_conv(float * state, const float * x, const float * w, float * out, int C, cudaStream_t s);
// beta = sigmoid(b), g = softplus(a + dt_bias) * ssm_a
void gdn_gates(const float * b_raw, const float * a_raw, const float * dt_bias, const float * ssm_a, float * beta,
               float * g, int H, cudaStream_t s);
// one recurrence step for H_v heads of S x S; v head h reads q/k head h % H_k. state[h][i][j] = S[i][j]
void gdn_step(float * state, const float * q, const float * k, const float * v, const float * g, const float * beta,
              float * out, int H_v, int H_k, int S, cudaStream_t s);

// QSA (build_layer_attn): [q|gate] per head -> q, gate
void split_q_gate(const float * q_full, float * q, float * gate, int heads, int D, cudaStream_t s);
void rope_neox(float * x, int heads, int D, int n_rot, int pos, float freq_base, cudaStream_t s);
// causal GQA attention of one query over n_kv cached positions; cache [n_kv][kv_heads][D]
void attention(const float * q, const float * kcache, const float * vcache, float * out, int heads, int kv_heads,
               int D, int n_kv, float scale, cudaStream_t s);

// MoE router: softmax over n_expert, top-k, weights / max(sum, 6.1e-5)  -> ids, w (device)
void router_topk(const float * logits, int n_expert, int k, int * ids, float * w, cudaStream_t s);
// the same for one or two routers in one launch (l1 may be null); n_expert <= 512
void router_topk2(const float * l0, int * ids0, float * w0, const float * l1, int * ids1, float * w1, int n_expert, int k,
                  cudaStream_t s);

// PLE (build_ple)
void ple_gate(const float * key_n, const float * query_n, float * gate, int D, int hc, cudaStream_t s);
void ple_gated_value(const float * value, const float * gate, float * gated, int D, int hc, cudaStream_t s);
// depthwise causal conv dilated by `dil` over the history (n_hist = (kern-1)*dil rows of C), silu; shifts history.
// out = silu(sum_k w[c][k] * x[t - (kern-1-k)*dil]);  hist: [n_hist][C], w: [C][kern]
void ple_conv(float * hist, const float * x, const float * w, float * out, int C, int kern, int dil, cudaStream_t s);

}  // namespace bl::cuda
