// Fused kernels of the decode path: fewer launches, the small matrices sharing a launch with the big ones.
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include "bl/cuda_pipeline.h"

namespace bl::cuda {

// Several matrices that read the same input x in one launch (one warp per output row across all segments).
// The input can be transformed on load: x, silu(x * xs), or silu(x) * x2 (SwiGLU of two vectors).
enum class Xform : int { None = 0, SiluScale = 1, SwiGLU = 2 };
constexpr int kMaxSegs = 8;
struct MvSeg {
    uint32_t        type;
    const void *    W;
    size_t          row_bytes;
    int             M;
    float *         y;
};
struct MvGroup {
    MvSeg seg[kMaxSegs];
    int   n = 0;
    int   K = 0;
    Xform xf = Xform::None;
    float xs = 1.f;
    const float * x2 = nullptr;
    bool  q8 = false;   // set by matmul_group: the dp4a segments read the activations as q8_1
    void add(uint32_t type, const void * W, size_t row_bytes, int M, float * y) { seg[n++] = {type, W, row_bytes, M, y}; }
};
void matvec_group(const MvGroup & g, const float * x, cudaStream_t s);
// the same for T tokens (T <= kMaxT): x is [T][K], segment outputs are [T][M] (and x2 [T][K] for SwiGLU).
// Every weight row is read once for all T tokens.
void matmul_group(const MvGroup & g, const float * x, int T, cudaStream_t s);
void set_matmul_q8(bool on);   // the dp4a path for quantized segments (default on; env BL_NO_Q8 turns it off)

// hyper-connection up projection + gated mean, fused:
//   gate[c*D+i] = W_up[c*D+i] . silu(lo / hc);  mixed[i] = (1/hc) sum_c xn[c*D+i] * sigmoid(gate[c*D+i])
void hc_up_mix(const void * W_up_bf16, const float * lo, int R, const float * xn, float * mixed, int D, int hc,
               cudaStream_t s);

// T-token versions: lo [T][R], xn [T][hc*D], mixed [T][D]
void hc_up_mix_t(const void * W_up_bf16, const float * lo, int R, const float * xn, float * mixed, int D, int hc, int T,
                 cudaStream_t s);
// R [T][hc*D] += h [T][D] * 2*sigmoid(inj [T][hc] / hc)
void hc_combine_t(float * R, const float * h, const float * inj, int D, int hc, int T, cudaStream_t s);
// out[t] = row tokens[t] of the embedding table; tokens in device memory
void embed_tokens(uint32_t type, const void * W, size_t row_bytes, int D, const int * tokens, int T, float * out,
                  cudaStream_t s);
// ids[t] = argmax of logits[t] (n values each), lowest id on ties; probs[t] = its softmax probability (if probs)
void argmax_rows(const float * logits, int n, int T, int * ids, cudaStream_t s, float * probs = nullptr);
// MTP input fusion: R[r][c] = h[r][c] + e[r]   (h [T][hc][D], e [T][D])
void add_broadcast_streams(const float * h, const float * e, float * R, int D, int hc, int T, cudaStream_t s);

// GDN after the projections: conv + silu (state shifts), L2 norm of q/k heads, beta/g gates.
//   qkv [Hk*S | Hk*S | Hv*S] -> conv_out (same layout, q/k normalized)
void gdn_pre(float * conv_state, const float * qkv, const float * conv_w, float * conv_out, int Hk, int Hv, int S,
             float eps, const float * b_raw, const float * a_raw, const float * dt_bias, const float * ssm_a,
             float * beta, float * g, cudaStream_t s);
// o = rmsnorm_per_head(o) * w * sigmoid(z)   (in place)
void gdn_post(float * o, const float * w, const float * z, int Hv, int S, float eps, cudaStream_t s);

// QSA after the projections: q (read from Wq's per-head [q | gate] layout) and k get RMSNorm + NEOX rope at `pos`;
// q goes to q_out [H][Dh], k and v are appended to the cache at `pos`
void qsa_pre(const float * q_full, float * q_out, int H, const float * k, const float * v, int Hkv, int Dh,
             const float * qn_w, const float * kn_w, float eps, int n_rot, float freq_base, const TokenState * ts,
             float * kcache, float * vcache, cudaStream_t s);
// attention whose output is gated: out[h] = attn(q[h]) * sigmoid(gate[h * gate_stride ...])
// n_kv = ts->pos + 1, at most max_kv (sizes the shared memory)
void attention_gated(const float * q, const float * kc, const float * vc, const float * gate, int gate_stride, float * out,
                     int heads, int kv_heads, int D, const TokenState * ts, int max_kv, float scale, cudaStream_t s);

// h = sum_i w[i] * exp_out[i] + sigmoid(*sg) * sh
void ffn_combine(const float * exp_out, const float * w, int n, const float * sh, const float * sg, float * h, int D,
                 cudaStream_t s);

// out = row ts->token of the embedding table (any supported format)
void embed_token(uint32_t type, const void * W, size_t row_bytes, int D, const TokenState * ts, float * out, cudaStream_t s);

}  // namespace bl::cuda
