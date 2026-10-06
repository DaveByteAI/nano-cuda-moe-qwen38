#include <cfloat>
#include <stdexcept>
#include <string>

#include "bl/cuda_window.h"
#include "kv.cuh"

namespace bl::cuda {

namespace {

constexpr int kMaxT = 4;

__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + expf(-x)); }
__device__ __forceinline__ float silu(float x) { return x / (1.f + expf(-x)); }

__device__ float block_sum(float v) {
    __shared__ float part[32];
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) part[w] = v;
    __syncthreads();
    v = l < static_cast<int>(blockDim.x >> 5) ? part[l] : 0.f;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ float block_max(float v) {
    __shared__ float part[32];
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) part[w] = v;
    __syncthreads();
    v = l < static_cast<int>(blockDim.x >> 5) ? part[l] : -FLT_MAX;
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// grid (2Hk + Hv + 1, T): blocks [0, 2Hk) q/k heads, [2Hk, 2Hk+Hv) v heads, the last the gates; blockIdx.y = token.
// Thread = channel within the head. The conv reads its 3 earlier inputs (from qkv, or the history st before the
// window), so tokens are independent (the same expressions as a per-channel loop over t, so the same bits).
__global__ void gdn_pre_t_k(const float * st, const float * qkv, const float * w, float * out, int Hk, int Hv, int S, float eps,
                            const float * b_raw, const float * a_raw, int abs, const float * dt, const float * A, float * beta,
                            float * g) {
    const int b = blockIdx.x, t = blockIdx.y;
    if (b == 2 * Hk + Hv) {
        for (int h = threadIdx.x; h < Hv; h += blockDim.x) {
            const int i = t * Hv + h;
            beta[i] = sigm(b_raw[t * abs + h]);
            const float z = a_raw[t * abs + h] + dt[h];
            g[i] = (z > 20.f ? z : logf(1.f + expf(z))) * A[h];
        }
        return;
    }
    const int C = (2 * Hk + Hv) * S;
    const int c = b * S + threadIdx.x;
    const float * wc = w + 4 * c;
    auto x_at = [&](int u) { return u >= 0 ? qkv[static_cast<size_t>(u) * C + c] : st[3 * c + u + 3]; };
    const float a0 = x_at(t - 3), a1 = x_at(t - 2), a2 = x_at(t - 1), xc = qkv[static_cast<size_t>(t) * C + c];
    const float v = silu(a0 * wc[0] + a1 * wc[1] + a2 * wc[2] + xc * wc[3]);
    if (b < 2 * Hk) {   // build_gdn_l2_norm: rms_norm(x, eps/n) / sqrt(n)
        const float ss = block_sum(v * v);
        out[static_cast<size_t>(t) * C + c] = v / sqrtf(ss / S + eps / S) * (1.f / sqrtf(static_cast<float>(S)));
    } else {
        out[static_cast<size_t>(t) * C + c] = v;
    }
}

__global__ void conv_commit_k(float * st, const float * x, int C, int H, const int * n_dev) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int n = *n_dev;
    if (n <= 0) return;
    // the last H of (state, x[0..n)): rows kept from the state first (in order, so the shift is in place), then x
    for (int r = 0; r < H; ++r) {
        const int src = n + r;   // index into the concatenation
        if (src < H) st[H * c + r] = st[H * c + src];
        else st[H * c + r] = x[static_cast<size_t>(src - H) * C + c];
    }
}

// The recurrence, per v head h: a block takes kGdnCols columns j of the S x S state, 4 threads per column (thread rq
// holds rows rq, rq + 4, ... in registers: the 4 read adjacent shared words). Per token: decay, sk = col . k,
// d = (v - sk) * beta, col += k d, o = col . q; the 4 partial sums of a column meet by shuffles. The next token's
// q, k (shared, double buffered) and v, beta, g (registers) are loaded while this one is computed.
constexpr int kGdnS = 128, kGdnCols = 32, kGdnQ = 4;
__global__ void __launch_bounds__(kGdnCols * kGdnQ) gdn_step_t_k(const float * sin, float * sout, const float * conv, int cd,
                                                                  const float * g, const float * beta, int gbs, float * out,
                                                                  int H_k, int T, const int * n_dev) {
    constexpr int S = kGdnS, R = S / kGdnQ, NB = S / kGdnCols;
    const int h = blockIdx.x / NB, tid = threadIdx.x, rq = tid & (kGdnQ - 1);
    const int j = (blockIdx.x % NB) * kGdnCols + tid / kGdnQ;
    const int n = n_dev ? *n_dev : T, Hv_total = gridDim.x / NB, hk = h % H_k;
    const float * si = sin + static_cast<size_t>(h) * S * S;
    float * so = sout + static_cast<size_t>(h) * S * S;
    __shared__ float qs[2][S], ks[2][S];
    float col[R];
#pragma unroll
    for (int ii = 0; ii < R; ++ii) col[ii] = si[(ii * kGdnQ + rq) * S + j];
    auto load_qk = [&](int t, int b) {   // blockDim == S
        const float * ct = conv + static_cast<size_t>(t) * cd;
        qs[b][tid] = ct[hk * S + tid];
        ks[b][tid] = ct[H_k * S + hk * S + tid];
    };
    float vn = 0.f, bn = 0.f, gn = 0.f;
    if (n > 0) {
        load_qk(0, 0);
        vn = conv[2 * H_k * S + h * S + j];
        bn = beta[h];
        gn = g[h];
    }
    __syncthreads();
    for (int t = 0; t < n; ++t) {
        const int b = t & 1;
        const float v = vn, be = bn, decay = expf(gn);
        if (t + 1 < n) {
            load_qk(t + 1, b ^ 1);
            const float * cn = conv + static_cast<size_t>(t + 1) * cd;
            vn = cn[2 * H_k * S + h * S + j];
            bn = beta[(t + 1) * gbs + h];
            gn = g[(t + 1) * gbs + h];
        }
        float sk = 0.f;
#pragma unroll
        for (int ii = 0; ii < R; ++ii) {
            col[ii] *= decay;
            sk += col[ii] * ks[b][ii * kGdnQ + rq];
        }
        sk += __shfl_xor_sync(0xffffffffu, sk, 1);
        sk += __shfl_xor_sync(0xffffffffu, sk, 2);
        const float d = (v - sk) * be;
        float o = 0.f;
#pragma unroll
        for (int ii = 0; ii < R; ++ii) {
            col[ii] += ks[b][ii * kGdnQ + rq] * d;
            o += col[ii] * qs[b][ii * kGdnQ + rq];
        }
        o += __shfl_xor_sync(0xffffffffu, o, 1);
        o += __shfl_xor_sync(0xffffffffu, o, 2);
        if (rq == 0) out[static_cast<size_t>(t) * Hv_total * S + h * S + j] = o * (1.f / sqrtf(static_cast<float>(S)));
        __syncthreads();   // buffer b ^ 1 is complete for t + 1; buffer b is free for t + 2
    }
#pragma unroll
    for (int ii = 0; ii < R; ++ii) so[(ii * kGdnQ + rq) * S + j] = col[ii];
}

__global__ void gdn_post_t_k(float * o, const float * w, const float * z, int S, float eps) {
    const int c = blockIdx.x * S + threadIdx.x;   // blocks: T * Hv
    const float v = o[c];
    const float ss = block_sum(v * v);
    o[c] = v / sqrtf(ss / S + eps) * w[threadIdx.x] * sigm(z[c]);
}

// grid (H + 2Hkv, T)
template <bool Q8>
__global__ void qsa_pre_t_k(const float * qf, float * qo, int H, const float * k, const float * v, int Hkv, int Dh, const float * qn,
                            const float * kn, float eps, int n_rot, float base, const TokenState * ts, KvCache kv) {
    extern __shared__ float row[];
    const int b = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    const int pos = ts->pos + t;
    if (b >= H + Hkv) {
        const int hv = b - H - Hkv;
        kv_st<Q8>(kv, 1, kv_row(kv, pos, Hkv, hv), d, v[(static_cast<size_t>(t) * Hkv + hv) * Dh + d]);
        return;
    }
    const bool isq = b < H;
    const float * x = isq ? qf + static_cast<size_t>(t) * 2 * H * Dh + b * 2 * Dh : k + (static_cast<size_t>(t) * Hkv + (b - H)) * Dh;
    const float val = x[d];
    const float ss = block_sum(val * val);
    row[d] = val / sqrtf(ss / Dh + eps) * (isq ? qn[d] : kn[d]);
    __syncthreads();
    const int half = n_rot / 2;
    float outv = row[d];
    if (d < n_rot) {
        const int i = d < half ? d : d - half;
        const float theta = pos * powf(base, -2.f * i / n_rot);
        float sn, cs;
        sincosf(theta, &sn, &cs);
        outv = d < half ? row[d] * cs - row[d + half] * sn : row[d - half] * sn + row[d] * cs;
    }
    if (isq) qo[(static_cast<size_t>(t) * H + b) * Dh + d] = outv;
    else kv_st<Q8>(kv, 0, kv_row(kv, pos, Hkv, b - H), d, outv);
}

// grid (heads, T)
// Split by position (flash-decoding): block (kv head, chunk of 32 positions) takes every query row on that kv head - its
// q heads for every token of the window - so a K/V chunk is read from memory once. Each warp takes rows; lane j
// scores position j of the chunk, each lane holds 8 of the 256 dims. Partial (max, sum, acc) per row and chunk, then
// merged by attn_merge_k. Grid sized for max_kv; chunks past a row's causal end are skipped.
constexpr int kAttC = 32, kAttMaxChunks = 72, kAttMaxRows = kMaxT * 32;
__device__ float  g_att_part[kAttMaxRows * kAttMaxChunks * 256];
__device__ float2 g_att_ml[kAttMaxRows * kAttMaxChunks];

// the cells query t attends: n of them, the j-th at position cell(j)
struct SelView {
    const int * l = nullptr;
    int lo = 0, n = 0;
    __device__ int cell(int j) const { return l ? l[j] : lo + j; }
};
__device__ __forceinline__ SelView sel_view(const AttnSel & sel, int t, int n_kv) {
    SelView v;
    if (sel.list) {
        v.l = sel.list + static_cast<size_t>(t) * sel.stride;
        v.n = sel.cnt[t];
    } else {
        v.lo = sel.window > 0 && n_kv > sel.window ? n_kv - sel.window : 0;
        v.n = n_kv - v.lo;
    }
    return v;
}
template <bool Q8>
__global__ void attn_chunk_k(const float * __restrict__ q, KvCache kv, int heads, int kv_heads, int T, const TokenState * ts, AttnSel sel,
                             float scale, int n_chunks, int per) {
    constexpr int D = 256;
    const int kvh = blockIdx.x, c = blockIdx.y, g = heads / kv_heads;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    const int pos = ts->pos;
    for (int r = warp; r < T * g; r += nw) {
        const int t = r / g, h = kvh * g + r % g;
        const SelView sv = sel_view(sel, t, pos + t + 1);
        if (c * per * kAttC >= sv.n) continue;
        float qv[8];
        {
            const float4 * qp = reinterpret_cast<const float4 *>(q + (static_cast<size_t>(t) * heads + h) * D) + 2 * lane;
            const float4 a = qp[0], b = qp[1];
            qv[0] = a.x; qv[1] = a.y; qv[2] = a.z; qv[3] = a.w; qv[4] = b.x; qv[5] = b.y; qv[6] = b.z; qv[7] = b.w;
        }
        float M = -FLT_MAX, L = 0.f, acc[8] = {};   // over the block's `per` chunks of kAttC positions
        for (int s = 0; s < per; ++s) {
            const int j0 = (c * per + s) * kAttC;
            if (j0 >= sv.n) break;
            const int np = min(kAttC, sv.n - j0);
            const int my_cell = lane < np ? sv.cell(j0 + lane) : 0;
            float mine = -FLT_MAX;
            for (int j = 0; j < np; ++j) {
                const int p = __shfl_sync(0xffffffffu, my_cell, j);
                float kf[8];
                kv_ld8<Q8>(kv, 0, kv_row(kv, p, kv_heads, kvh), lane, kf);
                float d = 0.f;
#pragma unroll
                for (int i = 0; i < 8; ++i) d += qv[i] * kf[i];
                for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
                if (lane == j) mine = d * scale;
            }
            float m = mine;
            for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
            const float e = lane < np ? expf(mine - m) : 0.f;
            float l = e;
            for (int o = 16; o > 0; o >>= 1) l += __shfl_xor_sync(0xffffffffu, l, o);
            float ac[8] = {};
            for (int j = 0; j < np; ++j) {
                const float ej = __shfl_sync(0xffffffffu, e, j);
                const int p = __shfl_sync(0xffffffffu, my_cell, j);
                float vv[8];
                kv_ld8<Q8>(kv, 1, kv_row(kv, p, kv_heads, kvh), lane, vv);
#pragma unroll
                for (int i = 0; i < 8; ++i) ac[i] += ej * vv[i];
            }
            if (s == 0) {   // (one chunk per block: exactly the chunk's own partials)
                M = m; L = l;
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[i] = ac[i];
            } else {
                const float nm = fmaxf(M, m), a = expf(M - nm), b = expf(m - nm);
                L = L * a + l * b;
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[i] = acc[i] * a + ac[i] * b;
                M = nm;
            }
        }
        const size_t idx = static_cast<size_t>(t * heads + h) * n_chunks + c;
        float4 * out = reinterpret_cast<float4 *>(g_att_part + idx * D) + 2 * lane;
        out[0] = make_float4(acc[0], acc[1], acc[2], acc[3]);
        out[1] = make_float4(acc[4], acc[5], acc[6], acc[7]);
        if (lane == 0) g_att_ml[idx] = make_float2(M, L);
    }
}

// grid T*heads, block 256 (one thread per dim): merge the row's chunks, divide, gate
__global__ void attn_merge_k(const float * gate, int gsh, int gst, float * out, int heads, const TokenState * ts, AttnSel sel,
                             int n_chunks, int per) {
    constexpr int D = 256;
    const int r = blockIdx.x, t = r / heads, h = r % heads, d = threadIdx.x;
    const SelView sv = sel_view(sel, t, ts->pos + t + 1);
    const int nc = (sv.n + per * kAttC - 1) / (per * kAttC);
    const float2 * ml = g_att_ml + static_cast<size_t>(r) * n_chunks;
    float M = -FLT_MAX;
    for (int c = 0; c < nc; ++c) M = fmaxf(M, ml[c].x);
    float num = 0.f, den = 0.f;
    for (int c = 0; c < nc; ++c) {
        const float w = expf(ml[c].x - M);
        den += w * ml[c].y;
        num += w * g_att_part[(static_cast<size_t>(r) * n_chunks + c) * D + d];
    }
    out[static_cast<size_t>(r) * D + d] = num / den * sigm(gate[static_cast<size_t>(t) * gst + h * gsh + d]);
}

// ---- the indexer
// NEOX rope of the first n_rot dims of row[] (Dh threads, one per dim), as qsa_pre_t_k does
__device__ __forceinline__ float rope_at(const float * row, int d, int n_rot, float base, int pos) {
    if (d >= n_rot) return row[d];
    const int half = n_rot / 2, i = d < half ? d : d - half;
    const float theta = pos * powf(base, -2.f * i / n_rot);
    float sn, cs;
    sincosf(theta, &sn, &cs);
    return d < half ? row[d] * cs - row[d + half] * sn : row[d - half] * sn + row[d] * cs;
}

// grid (Hq + 1, T), Di threads: the q heads normed + roped; the last block stores the raw key
__global__ void idx_prep_k(const float * q_raw, int qs, const float * k_raw, int ks, const float * qn, float eps, int n_rot,
                           float base, const TokenState * ts, int Hq, int Di, float * q_out, float * kraw) {
    __shared__ float row[256];
    const int b = blockIdx.x, t = blockIdx.y, d = threadIdx.x, pos = ts->pos + t;
    if (b == Hq) {
        kraw[static_cast<size_t>(pos % kIdxRing) * Di + d] = k_raw[static_cast<size_t>(t) * ks + d];
        return;
    }
    const float x = q_raw[static_cast<size_t>(t) * qs + b * Di + d];
    const float ss = block_sum(x * x);
    row[d] = x / sqrtf(ss / Di + eps) * qn[d];
    __syncthreads();
    q_out[(static_cast<size_t>(t) * Hq + b) * Di + d] = rope_at(row, d, n_rot, base, pos);
}

// grid T, Di threads: the block that position pos + t completes (if it completes one)
__global__ void idx_blocks_k(const float * kraw, float * kblk, const float * kn, float eps, int n_rot, float base,
                             const TokenState * ts, int Di) {
    __shared__ float row[256];
    const int p = ts->pos + blockIdx.x, d = threadIdx.x;
    if ((p + 1) % 4) return;
    const int b = p / 4;
    const float * r = kraw + static_cast<size_t>((4 * b) % kIdxRing) * Di;   // the 4 rows are adjacent (kIdxRing % 4 == 0)
    const float x = 0.25f * (r[d] + r[Di + d] + r[2 * Di + d] + r[3 * Di + d]);
    const float ss = block_sum(x * x);
    row[d] = x / sqrtf(ss / Di + eps) * kn[d];
    __syncthreads();
    kblk[static_cast<size_t>(b) * Di + d] = rope_at(row, d, n_rot, base, 4 * b);
}

__device__ int block_sum_i(int v) {
    __shared__ int part[32];
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) part[w] = v;
    __syncthreads();
    v = l < static_cast<int>(blockDim.x >> 5) ? part[l] : 0;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ int block_excl_scan(int v) {   // exclusive prefix sum over the block's threads in order
    __shared__ int part[32];
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    int x = v;
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if (l >= o) x += y;
    }
    __syncthreads();
    if (l == 31) part[w] = x;
    __syncthreads();
    if (w == 0) {
        int z = l < static_cast<int>(blockDim.x >> 5) ? part[l] : 0;
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, z, o);
            if (l >= o) z += y;
        }
        part[l] = z;   // inclusive over warps
    }
    __syncthreads();
    return x - v + (w > 0 ? part[w - 1] : 0);
}

// grid (ceil(max_blocks / 256), T): one thread per block key, the query's heads in shared memory
__global__ void idx_score_k(const float * qi, const float * kblk, const TokenState * ts, int Hq, int Di, int K, float * score,
                            int max_blocks) {
    const int t = blockIdx.y, q = ts->pos + t, nb = (q + 1) / 4;
    if (nb <= K || blockIdx.x * blockDim.x >= nb) return;   // nothing to rank, or past the complete blocks
    __shared__ float qs[1024];
    for (int i = threadIdx.x; i < Hq * Di; i += blockDim.x) qs[i] = qi[static_cast<size_t>(t) * Hq * Di + i];
    __syncthreads();
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nb) return;
    const float4 * kb = reinterpret_cast<const float4 *>(kblk + static_cast<size_t>(b) * Di);
    float acc[8] = {};
    for (int d4 = 0; d4 < Di / 4; ++d4) {
        const float4 k4 = __ldg(kb + d4);
#pragma unroll
        for (int h = 0; h < 8; ++h)
            if (h < Hq) {
                const float * qh = qs + h * Di + 4 * d4;
                acc[h] += qh[0] * k4.x + qh[1] * k4.y + qh[2] * k4.z + qh[3] * k4.w;
            }
    }
    float s = 0.f;
#pragma unroll
    for (int h = 0; h < 8; ++h)
        if (h < Hq) s += fmaxf(acc[h], 0.f);
    score[static_cast<size_t>(t) * max_blocks + b] = s;
}

// The same scores, with each block key read from memory once per call: thread i holds key b = 128 * blockIdx.x + i
// in registers and scores it against every query, the queries kIdxQg at a time in shared memory (read as broadcasts).
// The old kernel reads every key once per query: 256 queries x 16.7 MB of keys at 128K (beyond the 6 MB L2).
// Same sums in the same order as idx_score_k, so the same bits. Hq = 4, Di = 128. grid.y splits the queries into
// groups of kIdxQper (more blocks for short contexts; each key is then read once per group).
constexpr int kIdxKeys = 128, kIdxQg = 16, kIdxQper = 64;
__global__ void __launch_bounds__(kIdxKeys) idx_score_tiled_k(const float * qi, const float * kblk, const TokenState * ts, int K,
                                                              float * score, int max_blocks, int T) {
    constexpr int Hq = 4, Di = 128, D4 = Di / 4;
    __shared__ float4 qs[kIdxQg * Hq * D4];   // 32 KB
    const int pos = ts->pos, nb_last = (pos + min(T, static_cast<int>(blockIdx.y + 1) * kIdxQper)) / 4;   // its last query's complete blocks
    if (blockIdx.x * kIdxKeys >= nb_last) return;      // (the grid is sized for max_blocks)
    const int b = blockIdx.x * kIdxKeys + threadIdx.x;
    float4 k[D4];
    {
        const float4 * kb = reinterpret_cast<const float4 *>(kblk + static_cast<size_t>(min(b, nb_last - 1)) * Di);
#pragma unroll
        for (int d4 = 0; d4 < D4; ++d4) k[d4] = __ldg(kb + d4);
    }
    const int t_end = min(T, static_cast<int>(blockIdx.y + 1) * kIdxQper);
    for (int t0 = blockIdx.y * kIdxQper; t0 < t_end; t0 += kIdxQg) {
        const int ng = min(kIdxQg, t_end - t0);
        __syncthreads();
        const float4 * src = reinterpret_cast<const float4 *>(qi + static_cast<size_t>(t0) * Hq * Di);
        for (int i = threadIdx.x; i < ng * Hq * D4; i += kIdxKeys) qs[i] = src[i];
        __syncthreads();
        for (int g = 0; g < ng; ++g) {
            const int t = t0 + g, nb = (pos + t + 1) / 4;
            if (nb <= K || b >= nb) continue;
            float acc[Hq] = {};
#pragma unroll
            for (int d4 = 0; d4 < D4; ++d4)
#pragma unroll
                for (int h = 0; h < Hq; ++h) {
                    const float4 qh = qs[(g * Hq + h) * D4 + d4];
                    acc[h] += qh.x * k[d4].x + qh.y * k[d4].y + qh.z * k[d4].z + qh.w * k[d4].w;
                }
            float sc = 0.f;
#pragma unroll
            for (int h = 0; h < Hq; ++h) sc += fmaxf(acc[h], 0.f);
            score[static_cast<size_t>(t) * max_blocks + b] = sc;
        }
    }
}

// grid T, 1024 threads
__global__ void idx_select_k(const float * qi, const float * kblk, const TokenState * ts, int Hq, int Di, int K, float * score,
                             int max_blocks, int * list, int stride, int * cnt) {
    const int t = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    const int q = ts->pos + t, nb = (q + 1) / 4, tail0 = 4 * nb;
    int * L = list + static_cast<size_t>(t) * stride;
    if (nb <= K) {   // every cell
        for (int i = tid; i <= q; i += nt) L[i] = i;
        if (tid == 0) cnt[t] = q + 1;
        return;
    }
    float * sc = score + static_cast<size_t>(t) * max_blocks;   // by idx_score_k
    const int seg = (nb + nt - 1) / nt, b0 = min(nb, tid * seg), b1 = min(nb, b0 + seg);
    // the K-th largest score, by radix select over its bits (scores are >= 0: the bits order like the values)
    __shared__ int hist[256];
    __shared__ unsigned s_prefix;
    __shared__ int s_rem;
    unsigned prefix = 0;
    int rem = K;   // how many still to find among the elements matching prefix
    for (int shift = 24; shift >= 0; shift -= 8) {
        const unsigned mask_hi = shift == 24 ? 0u : ~((1u << (shift + 8)) - 1);
        for (int i = tid; i < 256; i += nt) hist[i] = 0;
        __syncthreads();
        for (int b = b0; b < b1; ++b) {
            const unsigned u = __float_as_uint(sc[b]);
            if ((u & mask_hi) == prefix) atomicAdd(&hist[(u >> shift) & 255], 1);
        }
        __syncthreads();
        if (tid == 0) {   // the digit where the count from the top reaches rem
            int acc = 0, dg = 255;
            for (; dg > 0; --dg) {
                if (acc + hist[dg] >= rem) break;
                acc += hist[dg];
            }
            s_prefix = prefix | (static_cast<unsigned>(dg) << shift);
            s_rem = rem - acc;
        }
        __syncthreads();
        prefix = s_prefix;
        rem = s_rem;
    }
    const unsigned lo = prefix;   // the K-th largest value's bits
    int gt = 0, eq = 0;
    for (int b = b0; b < b1; ++b) {
        const unsigned u = __float_as_uint(sc[b]);
        gt += u > lo;
        eq += u == lo;
    }
    const int need = K - block_sum_i(gt);   // ties at the threshold: the lowest blocks
    int eq_before = block_excl_scan(eq);
    int mine = 0;
    for (int b = b0; b < b1; ++b) {
        const unsigned u = __float_as_uint(sc[b]);
        mine += u > lo || (u == lo && eq_before++ < need);
    }
    int at = block_excl_scan(mine);
    eq_before = block_excl_scan(eq);
    for (int b = b0; b < b1; ++b) {
        const unsigned u = __float_as_uint(sc[b]);
        if (u > lo || (u == lo && eq_before++ < need)) {
            for (int j = 0; j < 4; ++j) L[4 * at + j] = 4 * b + j;
            ++at;
            sc[b] = -1.f;   // taken
        }
    }
    for (int i = tid; i <= q - tail0; i += nt) L[4 * K + i] = tail0 + i;
    // the reference's top-k is 4K + 3 cells wide: a tail shorter than 3 leaves room for the first cells of the next best
    // block (its lowest positions)
    const int extra = 3 - (q + 1 - tail0);
    float bv = -1.f;
    int bi = 0x7fffffff;
    for (int b = b0; b < b1; ++b)
        if (sc[b] > bv) { bv = sc[b]; bi = b; }
    __shared__ float wv[32];
    __shared__ int wi[32];
    for (int o = 16; o > 0; o >>= 1) {
        const float v2 = __shfl_xor_sync(0xffffffffu, bv, o);
        const int i2 = __shfl_xor_sync(0xffffffffu, bi, o);
        if (v2 > bv || (v2 == bv && i2 < bi)) { bv = v2; bi = i2; }
    }
    if ((tid & 31) == 0) { wv[tid >> 5] = bv; wi[tid >> 5] = bi; }
    __syncthreads();
    if (tid == 0) {
        for (int w = 1; w < nt / 32; ++w)
            if (wv[w] > bv || (wv[w] == bv && wi[w] < bi)) { bv = wv[w]; bi = wi[w]; }
        const int n_extra = bv >= 0.f ? extra : 0;
        for (int j = 0; j < n_extra; ++j) L[4 * K + (q + 1 - tail0) + j] = 4 * bi + j;
        cnt[t] = 4 * K + (q + 1 - tail0) + n_extra;
    }
}

// one block per token: the same rounds as router_topk2_k
__global__ void router_topk_t_k(const float * logits_all, int n, int k, int * ids_all, float * w_all) {
    const float * logits = logits_all + static_cast<size_t>(blockIdx.x) * n;
    int * ids = ids_all + blockIdx.x * k;
    float * w = w_all + blockIdx.x * k;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const float v = t < n ? logits[t] : -FLT_MAX;
    const float m = block_max(v);
    const float e = t < n ? expf(v - m) : 0.f;
    const float s = block_sum(e);
    float p = t < n ? e / s : -FLT_MAX;
    __shared__ float bv[32];
    __shared__ int bi[32];
    __shared__ int win;
    for (int r = 0; r < k; ++r) {
        float bvv = p;
        int bii = t;
        for (int o = 16; o > 0; o >>= 1) {
            const float v2 = __shfl_xor_sync(0xffffffffu, bvv, o);
            const int i2 = __shfl_xor_sync(0xffffffffu, bii, o);
            if (v2 > bvv || (v2 == bvv && i2 < bii)) { bvv = v2; bii = i2; }
        }
        if (lane == 0) { bv[warp] = bvv; bi[warp] = bii; }
        __syncthreads();
        if (warp == 0) {
            const int nw = blockDim.x >> 5;
            float x = lane < nw ? bv[lane] : -FLT_MAX;
            int xi = lane < nw ? bi[lane] : 1 << 30;
            for (int o = 16; o > 0; o >>= 1) {
                const float v2 = __shfl_xor_sync(0xffffffffu, x, o);
                const int i2 = __shfl_xor_sync(0xffffffffu, xi, o);
                if (v2 > x || (v2 == x && i2 < xi)) { x = v2; xi = i2; }
            }
            if (lane == 0) { win = xi; ids[r] = xi; w[r] = x; }
        }
        __syncthreads();
        if (t == win) p = -FLT_MAX;
    }
    __syncthreads();
    if (t == 0) {
        float sum = 0.f;
        for (int r = 0; r < k; ++r) sum += w[r];
        const float inv = 1.f / fmaxf(sum, 6.103515625e-5f);
        for (int r = 0; r < k; ++r) w[r] *= inv;
    }
}

__global__ void ple_conv_t_k(const float * hist, const float * x, const float * w, float * out, int C, int kern, int dil, int T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int n_hist = (kern - 1) * dil;
    for (int t = 0; t < T; ++t) {
        float acc = 0.f;
        for (int k = 0; k < kern; ++k) {
            const int src = t - (kern - 1 - k) * dil;   // window index; < 0 reads the history
            const float v = src >= 0 ? x[static_cast<size_t>(src) * C + c] : hist[static_cast<size_t>(n_hist + src) * C + c];
            acc += w[c * kern + k] * v;
        }
        out[static_cast<size_t>(t) * C + c] = silu(acc);
    }
}

__global__ void ple_commit_k(float * hist, const float * x, int C, int n_hist, const int * n_dev) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int n = *n_dev;
    if (n <= 0) return;
    // the last n_hist rows of (hist, x[0..n)), shifted in place
    for (int r = 0; r < n_hist; ++r) {
        const int src = n + r;
        hist[static_cast<size_t>(r) * C + c] = src < n_hist ? hist[static_cast<size_t>(src) * C + c] : x[static_cast<size_t>(src - n_hist) * C + c];
    }
}

}  // namespace

void gdn_pre_t(const float * st, const float * qkv, const float * w, float * out, int Hk, int Hv, int S, float eps, const float * b_raw,
               const float * a_raw, int abs, const float * dt, const float * A, float * beta, float * g, int T, cudaStream_t s) {
    if (T <= 0) return;
    gdn_pre_t_k<<<dim3(2 * Hk + Hv + 1, T), S, 0, s>>>(st, qkv, w, out, Hk, Hv, S, eps, b_raw, a_raw, abs, dt, A, beta, g);
}
void conv_commit(float * st, const float * x, int C, int H, const int * n_dev, cudaStream_t s) {
    conv_commit_k<<<(C + 255) / 256, 256, 0, s>>>(st, x, C, H, n_dev);
}
void gdn_step_t(const float * sin, float * sout, const float * conv, int cd, const float * g, const float * beta, int gbs, float * out,
                int H_v, int H_k, int S, int T, const int * n_dev, cudaStream_t s) {
    if (S != kGdnS) throw std::runtime_error("gdn_step_t: state size " + std::to_string(S));
    gdn_step_t_k<<<H_v * (kGdnS / kGdnCols), kGdnCols * kGdnQ, 0, s>>>(sin, sout, conv, cd, g, beta, gbs, out, H_k, T, n_dev);
}
void gdn_post_t(float * o, const float * w, const float * z, int Hv, int S, float eps, int T, cudaStream_t s) {
    gdn_post_t_k<<<Hv * T, S, 0, s>>>(o, w, z, S, eps);
}
void qsa_pre_t(const float * qf, float * qo, int H, const float * k, const float * v, int Hkv, int Dh, const float * qn, const float * kn,
               float eps, int n_rot, float base, const TokenState * ts, const KvCache & kv, int T, cudaStream_t s) {
    if (Dh != kKvD) throw std::runtime_error("qsa_pre_t: head dim " + std::to_string(Dh));
    auto kern = kv.sk ? qsa_pre_t_k<true> : qsa_pre_t_k<false>;
    kern<<<dim3(H + 2 * Hkv, T), Dh, static_cast<size_t>(Dh) * sizeof(float), s>>>(qf, qo, H, k, v, Hkv, Dh, qn, kn, eps, n_rot, base,
                                                                                   ts, kv);
}
void attention_t(const float * q, const KvCache & kv, const float * gate, int gsh, int gst, float * out, int heads,
                 int kv_heads, int D, const TokenState * ts, const AttnSel & sel, float scale, int T, cudaStream_t s) {
    // chunks of kAttC positions; past kAttMaxChunks of them, a block takes `per` consecutive chunks
    const int need = (sel.max_cells + kAttC - 1) / kAttC, per = (need + kAttMaxChunks - 1) / kAttMaxChunks;
    const int n_chunks = (need + per - 1) / per;
    if (D != 256 || T * heads > kAttMaxRows || heads % kv_heads)
        throw std::runtime_error("attention_t: unsupported shape (head dim " + std::to_string(D) + ", " +
                                 std::to_string(sel.max_cells) + " cells)");
    auto kern = kv.sk ? attn_chunk_k<true> : attn_chunk_k<false>;
    kern<<<dim3(kv_heads, n_chunks), 256, 0, s>>>(q, kv, heads, kv_heads, T, ts, sel, scale, n_chunks, per);
    attn_merge_k<<<T * heads, D, 0, s>>>(gate, gsh, gst, out, heads, ts, sel, n_chunks, per);
}
void idx_prep(const float * q_raw, int qs, const float * k_raw, int ks, const float * qn, float eps, int n_rot, float base,
              const TokenState * ts, int Hq, int Di, float * q_out, float * kraw, int T, cudaStream_t s) {
    idx_prep_k<<<dim3(Hq + 1, T), Di, 0, s>>>(q_raw, qs, k_raw, ks, qn, eps, n_rot, base, ts, Hq, Di, q_out, kraw);
}
void idx_blocks(const float * kraw, float * kblk, const float * kn, float eps, int n_rot, float base, const TokenState * ts, int Di,
                int T, cudaStream_t s) {
    idx_blocks_k<<<T, Di, 0, s>>>(kraw, kblk, kn, eps, n_rot, base, ts, Di);
}
void idx_select(const float * q, const float * kblk, const TokenState * ts, int Hq, int Di, int K, float * score, int max_blocks,
                int * list, int stride, int * cnt, int T, cudaStream_t s) {
    if (Hq * Di > 1024 || Hq > 8 || Di % 4) throw std::runtime_error("idx_select: indexer shape");
    if (Hq == 4 && Di == 128)
        idx_score_tiled_k<<<dim3((max_blocks + kIdxKeys - 1) / kIdxKeys, (T + kIdxQper - 1) / kIdxQper), kIdxKeys, 0, s>>>(
            q, kblk, ts, K, score, max_blocks, T);
    else
        idx_score_k<<<dim3((max_blocks + 255) / 256, T), 256, 0, s>>>(q, kblk, ts, Hq, Di, K, score, max_blocks);
    idx_select_k<<<T, 1024, 0, s>>>(q, kblk, ts, Hq, Di, K, score, max_blocks, list, stride, cnt);
}
void idx_select_ref(const float * q, const float * kblk, const TokenState * ts, int Hq, int Di, int K, float * score, int max_blocks,
                    int * list, int stride, int * cnt, int T, cudaStream_t s) {
    if (Hq * Di > 1024 || Hq > 8 || Di % 4) throw std::runtime_error("idx_select: indexer shape");
    idx_score_k<<<dim3((max_blocks + 255) / 256, T), 256, 0, s>>>(q, kblk, ts, Hq, Di, K, score, max_blocks);
    idx_select_k<<<T, 1024, 0, s>>>(q, kblk, ts, Hq, Di, K, score, max_blocks, list, stride, cnt);
}

void router_topk_t(const float * logits, int n, int k, int * ids, float * w, int T, cudaStream_t s) {
    if (n > 512) throw std::runtime_error("router_topk_t: more than 512 experts");
    router_topk_t_k<<<T, 512, 0, s>>>(logits, n, k, ids, w);
}
void ple_conv_t(const float * hist, const float * x, const float * w, float * out, int C, int kern, int dil, int T, cudaStream_t s) {
    ple_conv_t_k<<<(C + 255) / 256, 256, 0, s>>>(hist, x, w, out, C, kern, dil, T);
}
void ple_commit(float * hist, const float * x, int C, int n_hist, const int * n_dev, cudaStream_t s) {
    ple_commit_k<<<(C + 255) / 256, 256, 0, s>>>(hist, x, C, n_hist, n_dev);
}

}  // namespace bl::cuda
