// Kernels of a verify window: T consecutive tokens through a layer at once, the recurrent state untouched until the
// host knows how many of them were accepted (then a commit applies exactly that many).
#pragma once

#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "bl/cuda_pipeline.h"

namespace bl::cuda {

// GDN, before the recurrence: per channel the 4-tap conv over [history(3) | x_0..x_{T-1}] + silu (the history is read,
// not written), then the L2 norm of every (token, q/k head), and the gates beta/g per (token, v head).
//   qkv [T][C], out [T][C], b_raw/a_raw [T][Hv], beta/g [T][Hv]
void gdn_pre_t(const float * conv_state, const float * qkv, const float * conv_w, float * out, int Hk, int Hv, int S,
               float eps, const float * b_raw, const float * a_raw, int ab_stride, const float * dt_bias,
               const float * ssm_a, float * beta, float * g, int T, cudaStream_t s);
// history <- the last 3 of [history | x_0..x_{n-1}] (n = *n_dev)
void conv_commit(float * conv_state, const float * qkv, int C, int kern_minus_1, const int * n_dev, cudaStream_t s);
// the recurrence for n tokens (n = *n_dev, or T when n_dev is null): state_in is read, the state after the last of them
// written to state_out; out [T][Hv*S] (rows past n untouched)
void gdn_step_t(const float * state_in, float * state_out, const float * conv_out, int conv_dim, const float * g,
                const float * beta, int gb_stride, float * out, int H_v, int H_k, int S, int T, const int * n_dev,
                cudaStream_t s);
// o [T][Hv*S] = rmsnorm per head * w * sigmoid(z [T][Hv*S])
void gdn_post_t(float * o, const float * w, const float * z, int Hv, int S, float eps, int T, cudaStream_t s);

// A layer's KV cache, rows [cell][kv head][Dh = 256]: fp16 values, or int8 values with one fp16 scale per 32 of them
// (sk/sv [cell][kv head][8], non-null). Position p lives in cell p & mask (mask -1: cell p; 2^n - 1: a ring).
struct KvCache {
    void *   k = nullptr;
    void *   v = nullptr;
    __half * sk = nullptr;
    __half * sv = nullptr;
    int      mask = -1;
};
constexpr int kKvGroup = 32;   // values per int8 scale

// QSA: q (from Wq's per-head [q | gate] rows, [T][2*H*Dh]) and k ([T][Hkv*Dh]) normed + roped at pos + t; q to
// q_out [T][H*Dh]; k, v appended to the cache at pos + t (pos = ts->pos)
void qsa_pre_t(const float * q_full, float * q_out, int H, const float * k, const float * v, int Hkv, int Dh,
               const float * qn_w, const float * kn_w, float eps, int n_rot, float freq_base, const TokenState * ts,
               const KvCache & kv, int T, cudaStream_t s);

// Which cells a query attends to: the positions list[t][0 .. cnt[t]) (the indexer's selection, ascending), or with
// list null the last `window` cells up to pos + t (window 0: all of them).
struct AttnSel {
    const int * list = nullptr;
    const int * cnt = nullptr;
    int         stride = 0;    // ints per list row
    int         window = 0;
    int         max_cells = 0; // the most cells any query attends (sizes the split)
};
// out [T][H*D] gated by sigmoid(gate [T] rows of q_full)
void attention_t(const float * q, const KvCache & kv, const float * gate, int gate_stride_head,
                 int gate_stride_tok, float * out, int heads, int kv_heads, int D, const TokenState * ts, const AttnSel & sel,
                 float scale, int T, cudaStream_t s);

// The QSA indexer (DeepSeek-style "lightning indexer" over blocks of 4 cells), for queries at pos + t:
//   idx_prep:   q heads [T][Hq][Di] = rope(rmsnorm(q_raw) * qn), at pos + t; raw keys k_raw [T][Di] stored in kraw, a
//               ring of kIdxRing rows (row (pos + t) % kIdxRing): a block needs only its own 4 (T <= kIdxRing - 3)
//   idx_blocks: every block that position pos + t completes: its key = rope(rmsnorm(mean of its 4 raw keys) * kn) at
//               the block's first position, into kblk[block]
//   idx_select: per query, the incomplete tail [4*floor((q+1)/4), q] plus the top_blocks complete blocks before it
//               by sum_h relu(q_h . key) (all of them when they fit); ascending positions into list[t], count cnt[t].
//               score: scratch [T][max_blocks]
constexpr int kIdxRing = 512;
void idx_prep(const float * q_raw, int q_stride, const float * k_raw, int k_stride, const float * qn_w, float eps, int n_rot,
              float freq_base, const TokenState * ts, int Hq, int Di, float * q_out, float * kraw, int T, cudaStream_t s);
void idx_blocks(const float * kraw, float * kblk, const float * kn_w, float eps, int n_rot, float freq_base, const TokenState * ts,
                int Di, int T, cudaStream_t s);
void idx_select(const float * q, const float * kblk, const TokenState * ts, int Hq, int Di, int top_blocks, float * score,
                int max_blocks, int * list, int stride, int * cnt, int T, cudaStream_t s);
// testing: the same with the first scoring kernel (every key read once per query)
void idx_select_ref(const float * q, const float * kblk, const TokenState * ts, int Hq, int Di, int top_blocks, float * score,
                    int max_blocks, int * list, int stride, int * cnt, int T, cudaStream_t s);

// T routers at once: logits [T][n] -> ids [T][k], w [T][k]
void router_topk_t(const float * logits, int n, int k, int * ids, float * w, int T, cudaStream_t s);

// PLE depthwise dilated conv over [history | x_0..x_{T-1}] + silu, history read only; commit keeps the last rows
void ple_conv_t(const float * hist, const float * x, const float * w, float * out, int C, int kern, int dil, int T,
                cudaStream_t s);
void ple_commit(float * hist, const float * x, int C, int n_hist, const int * n_dev, cudaStream_t s);

}  // namespace bl::cuda
