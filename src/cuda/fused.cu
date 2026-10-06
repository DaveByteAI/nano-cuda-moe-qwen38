#include <cfloat>
#include <cstdlib>
#include <type_traits>
#include <stdexcept>
#include <string>

#include "bl/cuda_fused.h"
#include "dequant.cuh"
#include "rowdot.cuh"
#include "dp4a.cuh"

namespace bl::cuda {

namespace {

__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + expf(-x)); }
__device__ __forceinline__ float silu(float x) { return x / (1.f + expf(-x)); }

template <int R>
__device__ void rows_dot_any(uint32_t t, const uint8_t * const * rows, const float * xs, int K, int lane, float * acc) {
    switch (t) {
        case F32:     return rows_dot_smem<F32, R>(rows, xs, K, lane, acc);
        case BF16:    return rows_dot_smem<BF16, R>(rows, xs, K, lane, acc);
        case Q8_0:    return rows_dot_smem<Q8_0, R>(rows, xs, K, lane, acc);
        case Q4_K:    return rows_dot_smem<Q4_K, R>(rows, xs, K, lane, acc);
        case Q5_K:    return rows_dot_smem<Q5_K, R>(rows, xs, K, lane, acc);
        case Q6_K:    return rows_dot_smem<Q6_K, R>(rows, xs, K, lane, acc);
        case IQ2_XXS: return rows_dot_smem<IQ2_XXS, R>(rows, xs, K, lane, acc);
        case IQ2_XS:  return rows_dot_smem<IQ2_XS, R>(rows, xs, K, lane, acc);
        case IQ2_S:   return rows_dot_smem<IQ2_S, R>(rows, xs, K, lane, acc);
        case IQ3_XXS: return rows_dot_smem<IQ3_XXS, R>(rows, xs, K, lane, acc);
        case IQ3_S:   return rows_dot_smem<IQ3_S, R>(rows, xs, K, lane, acc);
        case IQ4_NL:  return rows_dot_smem<IQ4_NL, R>(rows, xs, K, lane, acc);
        case IQ4_XS:  return rows_dot_smem<IQ4_XS, R>(rows, xs, K, lane, acc);
        case Q2_0:    return rows_dot_smem<Q2_0, R>(rows, xs, K, lane, acc);
        default:      return;
    }
}

constexpr int kWarps = 8;
constexpr int kMaxK8 = 12288;   // the widest K the q8_1 path quantizes

__device__ __forceinline__ void stage_x(const MvGroup & g, const float * __restrict__ x, float * xs) {
    const int G = g.K / 8;
    for (int i = threadIdx.x; i < g.K; i += blockDim.x) {
        float v = x[i];
        if (g.xf == Xform::SiluScale) v = silu(v * g.xs);
        else if (g.xf == Xform::SwiGLU) v = silu(v) * g.x2[i];
        xs[(i & 7) * G + (i >> 3)] = v;
    }
    __syncthreads();
}

__device__ __forceinline__ int seg_of(const MvGroup & g, int row, int & r) {
    int s = 0, base = 0;
    while (row >= base + g.seg[s].M) base += g.seg[s++].M;
    r = row - base;
    return s;
}

// many rows: one warp per row
__global__ void mv_group_k(MvGroup g, const float * __restrict__ x, int total_rows) {
    extern __shared__ float xs[];
    stage_x(g, x, xs);
    const int lane = threadIdx.x & 31;
    for (int w = blockIdx.x * kWarps + (threadIdx.x >> 5); w < total_rows; w += gridDim.x * kWarps) {
        int r;
        const MvSeg & sg = g.seg[seg_of(g, w, r)];
        const uint8_t * rows[1] = {static_cast<const uint8_t *>(sg.W) + static_cast<size_t>(r) * sg.row_bytes};
        float acc = 0.f;
        rows_dot_any<1>(sg.type, rows, xs, g.K, lane, &acc);
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) sg.y[r] = acc;
    }
}

// few rows (long K): one block per row, its warps split K, then a shared-memory reduction
__device__ __forceinline__ void rows_dot_split(uint32_t t, const uint8_t * row, const float * xs, int K, float & acc);
template <int T>
__device__ __forceinline__ void split_dot(const uint8_t * row, const float * xs, int K, float & acc) {
    const int G = K / 8;
    for (int gg = threadIdx.x; gg < G; gg += blockDim.x) {
        float v[8];
        dequant8<T>(row + static_cast<size_t>(gg / Grp<T>::groups) * Grp<T>::bytes, gg % Grp<T>::groups, v);
#pragma unroll
        for (int j = 0; j < 8; ++j) acc += v[j] * xs[j * G + gg];
    }
}
__device__ __forceinline__ void rows_dot_split(uint32_t t, const uint8_t * row, const float * xs, int K, float & acc) {
    switch (t) {
        case F32:     return split_dot<F32>(row, xs, K, acc);
        case BF16:    return split_dot<BF16>(row, xs, K, acc);
        case Q8_0:    return split_dot<Q8_0>(row, xs, K, acc);
        case Q4_K:    return split_dot<Q4_K>(row, xs, K, acc);
        case Q5_K:    return split_dot<Q5_K>(row, xs, K, acc);
        case Q6_K:    return split_dot<Q6_K>(row, xs, K, acc);
        case IQ2_XXS: return split_dot<IQ2_XXS>(row, xs, K, acc);
        case IQ2_XS:  return split_dot<IQ2_XS>(row, xs, K, acc);
        case IQ2_S:   return split_dot<IQ2_S>(row, xs, K, acc);
        case IQ3_XXS: return split_dot<IQ3_XXS>(row, xs, K, acc);
        case IQ3_S:   return split_dot<IQ3_S>(row, xs, K, acc);
        case IQ4_NL:  return split_dot<IQ4_NL>(row, xs, K, acc);
        case IQ4_XS:  return split_dot<IQ4_XS>(row, xs, K, acc);
        case Q2_0:    return split_dot<Q2_0>(row, xs, K, acc);
        default:      return;
    }
}

__global__ void mv_group_split_k(MvGroup g, const float * __restrict__ x, int total_rows) {
    extern __shared__ float xs[];
    __shared__ float part[kWarps];
    stage_x(g, x, xs);
    for (int row = blockIdx.x; row < total_rows; row += gridDim.x) {
        int r;
        const MvSeg & sg = g.seg[seg_of(g, row, r)];
        float acc = 0.f;
        rows_dot_split(sg.type, static_cast<const uint8_t *>(sg.W) + static_cast<size_t>(r) * sg.row_bytes, xs, g.K, acc);
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = acc;
        __syncthreads();
        if (threadIdx.x == 0) {
            float t = 0.f;
            for (int w = 0; w < kWarps; ++w) t += part[w];
            sg.y[r] = t;
        }
        __syncthreads();
    }
}

// one warp per output i: the hc rows c*D+i, their sigmoid gates times xn, averaged
__global__ void hc_up_mix_k(const uint16_t * __restrict__ W, const float * __restrict__ lo, int R, const float * __restrict__ xn,
                            float * __restrict__ mixed, int D, int hc) {
    extern __shared__ float ls[];
    for (int i = threadIdx.x; i < R; i += blockDim.x) ls[i] = silu(lo[i] / hc);
    __syncthreads();
    const int lane = threadIdx.x & 31;
    for (int i = blockIdx.x * kWarps + (threadIdx.x >> 5); i < D; i += gridDim.x * kWarps) {
        float acc_mix = 0.f;
        for (int c = 0; c < hc; ++c) {
            const uint16_t * row = W + static_cast<size_t>(c * D + i) * R;
            float acc = 0.f;
            for (int k = lane; k < R; k += 32) acc += __uint_as_float(static_cast<uint32_t>(row[k]) << 16) * ls[k];
            for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
            acc_mix += xn[c * D + i] * sigm(acc);
        }
        if (lane == 0) mixed[i] = acc_mix * (1.f / hc);
    }
}

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

// blocks [0, 2Hk): q/k heads (conv + silu + L2 norm); [2Hk, 2Hk+Hv): v heads (conv + silu); last: gates
__global__ void gdn_pre_k(float * state, const float * qkv, const float * w, float * out, int Hk, int Hv, int S, float eps,
                          const float * b_raw, const float * a_raw, const float * dt, const float * A, float * beta, float * g) {
    const int b = blockIdx.x;
    if (b == 2 * Hk + Hv) {
        for (int h = threadIdx.x; h < Hv; h += blockDim.x) {
            beta[h] = sigm(b_raw[h]);
            const float z = a_raw[h] + dt[h];
            g[h] = (z > 20.f ? z : logf(1.f + expf(z))) * A[h];
        }
        return;
    }
    const int c = b * S + threadIdx.x;   // channel
    float * st = state + 3 * c;
    const float * wc = w + 4 * c;
    const float xc = qkv[c];
    const float v = silu(st[0] * wc[0] + st[1] * wc[1] + st[2] * wc[2] + xc * wc[3]);
    st[0] = st[1]; st[1] = st[2]; st[2] = xc;
    if (b < 2 * Hk) {   // build_gdn_l2_norm: rms_norm(x, eps/n) / sqrt(n)
        const float ss = block_sum(v * v);
        out[c] = v / sqrtf(ss / S + eps / S) * (1.f / sqrtf(static_cast<float>(S)));
    } else {
        out[c] = v;
    }
}

__global__ void gdn_post_k(float * o, const float * w, const float * z, int S, float eps) {
    const int c = blockIdx.x * S + threadIdx.x;
    const float v = o[c];
    const float ss = block_sum(v * v);
    o[c] = v / sqrtf(ss / S + eps) * w[threadIdx.x] * sigm(z[c]);
}

// blocks [0,H): q heads; [H,H+Hkv): k heads (normed, roped, cached); [H+Hkv, H+2Hkv): v heads (cached)
__global__ void qsa_pre_k(const float * qf, float * qo, int H, const float * k, const float * v, int Hkv, int Dh,
                          const float * qn, const float * kn, float eps, int n_rot, float base, const TokenState * ts,
                          float * kc, float * vc) {
    extern __shared__ float row[];
    const int pos = ts->pos;
    const int b = blockIdx.x, d = threadIdx.x;
    if (b >= H + Hkv) {
        const int hv = b - H - Hkv;
        vc[(static_cast<size_t>(pos) * Hkv + hv) * Dh + d] = v[hv * Dh + d];
        return;
    }
    const bool isq = b < H;
    const float * x = isq ? qf + b * 2 * Dh : k + (b - H) * Dh;
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
    if (isq) qo[b * Dh + d] = outv;
    else kc[(static_cast<size_t>(pos) * Hkv + (b - H)) * Dh + d] = outv;
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

__global__ void attention_gated_k(const float * q, const float * kc, const float * vc, const float * gate, int gstride,
                                  float * out, int heads, int kv_heads, int D, const TokenState * ts, float scale) {
    extern __shared__ float sc[];
    const int n_kv = ts->pos + 1;
    const int h = blockIdx.x, kvh = h / (heads / kv_heads);
    const float * qh = q + h * D;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    for (int p = warp; p < n_kv; p += nw) {
        const float * kp = kc + (static_cast<size_t>(p) * kv_heads + kvh) * D;
        float dot = 0.f;
        for (int d = lane; d < D; d += 32) dot += qh[d] * kp[d];
        for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
        if (lane == 0) sc[p] = dot * scale;
    }
    __syncthreads();
    float m = -FLT_MAX;
    for (int p = threadIdx.x; p < n_kv; p += blockDim.x) m = fmaxf(m, sc[p]);
    m = block_max(m);
    float sum = 0.f;
    for (int p = threadIdx.x; p < n_kv; p += blockDim.x) {
        const float e = expf(sc[p] - m);
        sc[p] = e;
        sum += e;
    }
    sum = block_sum(sum);
    __syncthreads();
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float acc = 0.f;
        for (int p = 0; p < n_kv; ++p) acc += sc[p] * vc[(static_cast<size_t>(p) * kv_heads + kvh) * D + d];
        out[h * D + d] = acc / sum * sigm(gate[h * gstride + d]);
    }
}

__global__ void ffn_combine_k(const float * eo, const float * w, int n, const float * sh, const float * sg, float * h, int D) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= D) return;
    float acc = 0.f;
    for (int e = 0; e < n; ++e) acc += w[e] * eo[static_cast<size_t>(e) * D + i];
    h[i] = acc + sigm(*sg) * sh[i];
}

// ---- T-token kernels ----
template <int T, int NT>
__device__ __forceinline__ void rows_dot_t(const uint8_t * row, const float * __restrict__ x, const float * __restrict__ x2,
                                           Xform xf, float xs, int K, int lane, float * acc, int step = 32) {
    for (int gg = lane; gg < K / 8; gg += step) {
        float v[8];
        dequant8<T>(row + static_cast<size_t>(gg / Grp<T>::groups) * Grp<T>::bytes, gg % Grp<T>::groups, v);
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const float4 a = reinterpret_cast<const float4 *>(x + static_cast<size_t>(t) * K)[2 * gg];
            const float4 b = reinterpret_cast<const float4 *>(x + static_cast<size_t>(t) * K)[2 * gg + 1];
            float xv[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
            if (xf != Xform::None) {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    if (xf == Xform::SiluScale) xv[j] = silu(xv[j] * xs);
                    else xv[j] = silu(xv[j]) * x2[static_cast<size_t>(t) * K + 8 * gg + j];
                }
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) acc[t] += v[j] * xv[j];
        }
    }
}

template <int NT>
__device__ void rows_dot_t_any(uint32_t type, const uint8_t * row, const float * x, const float * x2, Xform xf, float xs, int K,
                               int lane, float * acc, int step = 32) {
    switch (type) {
        case F32:     return rows_dot_t<F32, NT>(row, x, x2, xf, xs, K, lane, acc, step);
        case BF16:    return rows_dot_t<BF16, NT>(row, x, x2, xf, xs, K, lane, acc, step);
        case Q4_K:    return rows_dot_t<Q4_K, NT>(row, x, x2, xf, xs, K, lane, acc, step);
        case Q5_K:    return rows_dot_t<Q5_K, NT>(row, x, x2, xf, xs, K, lane, acc, step);
        case Q6_K:    return rows_dot_t<Q6_K, NT>(row, x, x2, xf, xs, K, lane, acc, step);
        case Q8_0:    return rows_dot_t<Q8_0, NT>(row, x, x2, xf, xs, K, lane, acc, step);
        case IQ2_XXS: return rows_dot_t<IQ2_XXS, NT>(row, x, x2, xf, xs, K, lane, acc, step);
        case IQ2_XS:  return rows_dot_t<IQ2_XS, NT>(row, x, x2, xf, xs, K, lane, acc, step);
        case IQ2_S:   return rows_dot_t<IQ2_S, NT>(row, x, x2, xf, xs, K, lane, acc, step);
        case IQ3_XXS: return rows_dot_t<IQ3_XXS, NT>(row, x, x2, xf, xs, K, lane, acc, step);
        case IQ3_S:   return rows_dot_t<IQ3_S, NT>(row, x, x2, xf, xs, K, lane, acc, step);
        case IQ4_NL:  return rows_dot_t<IQ4_NL, NT>(row, x, x2, xf, xs, K, lane, acc, step);
        case IQ4_XS:  return rows_dot_t<IQ4_XS, NT>(row, x, x2, xf, xs, K, lane, acc, step);
        case Q2_0:    return rows_dot_t<Q2_0, NT>(row, x, x2, xf, xs, K, lane, acc, step);
        default:      return;
    }
}

// many rows: a warp per row, all NT tokens
// BF16 groups with few rows and long K (the hyper-connection down/inject: 324 rows of 10240): a block takes 8 rows
// (a warp each) and one slice of K, with the slice of every token's x staged in shared memory once for those rows;
// the slices' partial sums go to g_bf_part and bf16_reduce_k adds them in a fixed order (no atomics: deterministic).
constexpr int kBfWarps = 8, kBfRowsPerWarp = 2, kBfRows = kBfWarps * kBfRowsPerWarp, kBfMaxSplit = 32, kBfMaxRows = 1024;
__device__ float g_bf_part[kBfMaxSplit * kMaxT * kBfMaxRows];
__device__ unsigned g_bf_done[kBfMaxRows / kBfRows + 1];   // per row tile: slices finished (the last one reduces)
template <int NT>
__global__ void bf16_splitk_k(MvGroup g, const float * __restrict__ x, int total_rows, int kc) {
    extern __shared__ float xs[];   // [NT][kc]
    const int s = blockIdx.y, k0 = s * kc, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int i = threadIdx.x; i < NT * kc / 4; i += blockDim.x) {
        const int t = i / (kc / 4), j = i % (kc / 4);
        reinterpret_cast<float4 *>(xs)[i] = reinterpret_cast<const float4 *>(x + static_cast<size_t>(t) * g.K + k0)[j];
    }
    __syncthreads();
    // a warp's rows (kBfRowsPerWarp, consecutive): each x value read from shared memory serves all of them
    const int row0 = (blockIdx.x * kBfWarps + warp) * kBfRowsPerWarp;
    if (row0 < total_rows) {
    const uint16_t * w[kBfRowsPerWarp];
#pragma unroll
    for (int q = 0; q < kBfRowsPerWarp; ++q) {
        const int row = min(row0 + q, total_rows - 1);   // a missing row repeats the last (its result is not stored)
        int r;
        const MvSeg & sg = g.seg[seg_of(g, row, r)];
        w[q] = static_cast<const uint16_t *>(sg.W) + static_cast<size_t>(r) * (sg.row_bytes / 2) + k0;
    }
    float acc[kBfRowsPerWarp][NT] = {};
    for (int k = lane * 8; k < kc; k += 256) {
        uint4 v[kBfRowsPerWarp];
#pragma unroll
        for (int q = 0; q < kBfRowsPerWarp; ++q) v[q] = *reinterpret_cast<const uint4 *>(w[q] + k);
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const float4 a = *reinterpret_cast<const float4 *>(xs + t * kc + k), b = *reinterpret_cast<const float4 *>(xs + t * kc + k + 4);
            const float xv[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
            for (int q = 0; q < kBfRowsPerWarp; ++q) {
                const uint32_t u[4] = {v[q].x, v[q].y, v[q].z, v[q].w};
                float sum = 0.f;
#pragma unroll
                for (int j = 0; j < 4; ++j) sum += __uint_as_float(u[j] << 16) * xv[2 * j] + __uint_as_float(u[j] & 0xffff0000u) * xv[2 * j + 1];
                acc[q][t] += sum;
            }
        }
    }
#pragma unroll
    for (int q = 0; q < kBfRowsPerWarp; ++q)
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            float a = acc[q][t];
            for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
            if (lane == 0 && row0 + q < total_rows) g_bf_part[(static_cast<size_t>(s) * NT + t) * total_rows + row0 + q] = a;
        }
    }
    // the tile's last slice to finish adds the slices up, in slice order
    __shared__ bool last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        last = atomicAdd(&g_bf_done[blockIdx.x], 1u) == gridDim.y - 1;
        if (last) g_bf_done[blockIdx.x] = 0;   // ready for the next call (nothing else touches it until then)
    }
    __syncthreads();
    if (!last) return;
    __threadfence();
    for (int i = threadIdx.x; i < NT * kBfRows; i += blockDim.x) {
        const int t = i / kBfRows, row = blockIdx.x * kBfRows + i % kBfRows;
        if (row >= total_rows) continue;
        float a = 0.f;
        for (int q = 0; q < static_cast<int>(gridDim.y); ++q) a += __ldcg(&g_bf_part[(static_cast<size_t>(q) * NT + t) * total_rows + row]);
        int r;
        const MvSeg & sg = g.seg[seg_of(g, row, r)];
        sg.y[static_cast<size_t>(t) * sg.M + r] = a;
    }
}
// the quantized formats against q8_1 activations (xq [NT][nb]): sub-blocks sb0, sb0 + step, ...
template <int NT>
__device__ void rows_dot_q8(uint32_t type, const uint8_t * row, const BlockQ8 * xq, int nb, int sb0, int step, float * acc) {
    switch (type) {
#define BL_CASE(F) case F: for (int sb = sb0; sb < nb; sb += step) dot32m<F, NT>(row, sb, xq, nb, NT, acc); return;
        BL_CASE(IQ2_XXS) BL_CASE(IQ2_XS) BL_CASE(IQ2_S) BL_CASE(IQ3_XXS) BL_CASE(IQ3_S) BL_CASE(IQ4_NL) BL_CASE(Q2_0)
        BL_CASE(Q4_K) BL_CASE(Q5_K) BL_CASE(Q6_K) BL_CASE(IQ4_XS)
#undef BL_CASE
        default: return;
    }
}

// the activations of a multi-token matmul as q8_1, transform applied: one warp per 32 values
__device__ BlockQ8 g_xq[kMaxT * kMaxK8 / 32];
__global__ void quantize_x_k(const float * x, const float * x2, Xform xf, float xs, int nblocks) {
    const int blk = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (blk >= nblocks) return;
    const size_t i = 32 * static_cast<size_t>(blk) + lane;
    float v = x[i];
    if (xf == Xform::SiluScale) v = silu(v * xs);
    else if (xf == Xform::SwiGLU) v = silu(v) * x2[i];
    float amax = fabsf(v), sum = v;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    }
    const float d = amax / 127.f;
    g_xq[blk].qs[lane] = amax == 0.f ? 0 : static_cast<int8_t>(roundf(v / d));
    if (lane == 0) g_xq[blk].ds = make_half2(__float2half(d), __float2half(sum));
}

template <int NT>
__global__ void mm_group_k(MvGroup g, const float * __restrict__ x, int total_rows) {
    const int lane = threadIdx.x & 31;
    for (int w = blockIdx.x * kWarps + (threadIdx.x >> 5); w < total_rows; w += gridDim.x * kWarps) {
        int r;
        const MvSeg & sg = g.seg[seg_of(g, w, r)];
        float acc[NT] = {};
        const uint8_t * row = static_cast<const uint8_t *>(sg.W) + static_cast<size_t>(r) * sg.row_bytes;
        if (g.q8 && has_dp4a(sg.type)) rows_dot_q8<NT>(sg.type, row, g_xq, g.K / 32, lane, 32, acc);
        else rows_dot_t_any<NT>(sg.type, row, x, g.x2, g.xf, g.xs, g.K, lane, acc);
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            float a = acc[t];
            for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
            if (lane == 0) sg.y[static_cast<size_t>(t) * sg.M + r] = a;
        }
    }
}

// few rows: a block per row, the warps split K
// LPR lanes per row (32 / LPR rows per warp): for K whose 32-value sub-blocks do not split evenly over 32 lanes
// (K = 2560: 80 of them, 2.5 per lane) 16 lanes take 5 each and no lane idles
template <int NT, int LPR>
__global__ void mm_group_lpr_k(MvGroup g, const float * __restrict__ x, int total_rows) {
    constexpr int RPW = 32 / LPR;
    const int lane = threadIdx.x & 31, sub = lane % LPR;
    for (int w = blockIdx.x * kWarps + (threadIdx.x >> 5); w * RPW < total_rows; w += gridDim.x * kWarps) {
        const int row = w * RPW + lane / LPR;
        const bool valid = row < total_rows;
        float acc[NT] = {};
        int r = 0;
        const MvSeg * sg = nullptr;
        if (valid) {
            sg = &g.seg[seg_of(g, row, r)];
            const uint8_t * rp = static_cast<const uint8_t *>(sg->W) + static_cast<size_t>(r) * sg->row_bytes;
            if (g.q8 && has_dp4a(sg->type)) rows_dot_q8<NT>(sg->type, rp, g_xq, g.K / 32, sub, LPR, acc);
            else rows_dot_t_any<NT>(sg->type, rp, x, g.x2, g.xf, g.xs, g.K, sub, acc, LPR);
        }
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            float a = acc[t];
            for (int o = LPR / 2; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
            if (valid && sub == 0) sg->y[static_cast<size_t>(t) * sg->M + r] = a;
        }
    }
}

template <int NT>
__global__ void mm_group_split_k(MvGroup g, const float * __restrict__ x, int total_rows) {
    __shared__ float part[kWarps][NT];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int G = g.K / 8;
    for (int row = blockIdx.x; row < total_rows; row += gridDim.x) {
        int r;
        const MvSeg & sg = g.seg[seg_of(g, row, r)];
        // the warps take interleaved stripes of 32 groups: present them to rows_dot_t as a K/8-group row slice
        float acc[NT] = {};
        const uint8_t * rowp = static_cast<const uint8_t *>(sg.W) + static_cast<size_t>(r) * sg.row_bytes;
        if (g.q8 && has_dp4a(sg.type)) rows_dot_q8<NT>(sg.type, rowp, g_xq, g.K / 32, warp * 32 + lane, 32 * kWarps, acc);
        // each warp handles groups gg = warp*32 + lane + 256*i, by calling the per-lane loop on a strided view
        else for (int base = warp * 32; base < G; base += 32 * kWarps) {
            const int gg = base + lane;
            if (gg < G) {
                float v[8];
                switch (sg.type) {
#define BL_CASE(F) case F: dequant8<F>(rowp + static_cast<size_t>(gg / Grp<F>::groups) * Grp<F>::bytes, gg % Grp<F>::groups, v); break;
                    BL_CASE(F32) BL_CASE(BF16) BL_CASE(Q4_K) BL_CASE(Q5_K) BL_CASE(Q6_K) BL_CASE(Q8_0) BL_CASE(IQ2_XXS)
                    BL_CASE(IQ2_XS) BL_CASE(IQ2_S) BL_CASE(IQ3_XXS) BL_CASE(IQ3_S) BL_CASE(IQ4_NL) BL_CASE(IQ4_XS) BL_CASE(Q2_0)
#undef BL_CASE
                    default: for (int j = 0; j < 8; ++j) v[j] = 0.f;
                }
#pragma unroll
                for (int t = 0; t < NT; ++t) {
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        float xv = x[static_cast<size_t>(t) * g.K + 8 * gg + j];
                        if (g.xf == Xform::SiluScale) xv = silu(xv * g.xs);
                        else if (g.xf == Xform::SwiGLU) xv = silu(xv) * g.x2[static_cast<size_t>(t) * g.K + 8 * gg + j];
                        acc[t] += v[j] * xv;
                    }
                }
            }
        }
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            float a = acc[t];
            for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
            if (lane == 0) part[warp][t] = a;
        }
        __syncthreads();
        if (threadIdx.x < NT) {
            float tsum = 0.f;
            for (int w = 0; w < kWarps; ++w) tsum += part[w][threadIdx.x];
            sg.y[static_cast<size_t>(threadIdx.x) * sg.M + r] = tsum;
        }
        __syncthreads();
    }
}

__global__ void hc_up_mix_t_k(const uint16_t * __restrict__ W, const float * __restrict__ lo, int R, const float * __restrict__ xn,
                              float * __restrict__ mixed, int D, int hc, int T) {
    extern __shared__ float ls[];   // [T][R]
    for (int i = threadIdx.x; i < T * R; i += blockDim.x) ls[i] = silu(lo[i] / hc);
    __syncthreads();
    const int lane = threadIdx.x & 31;
    for (int i = blockIdx.x * kWarps + (threadIdx.x >> 5); i < D; i += gridDim.x * kWarps) {
        float mix[kMaxT] = {};
        for (int c = 0; c < hc; ++c) {
            const uint16_t * row = W + static_cast<size_t>(c * D + i) * R;
            float acc[kMaxT] = {};
            for (int k = lane; k < R; k += 32) {
                const float w = __uint_as_float(static_cast<uint32_t>(row[k]) << 16);
                for (int t = 0; t < T; ++t) acc[t] += w * ls[t * R + k];
            }
            for (int t = 0; t < T; ++t) {
                float a = acc[t];
                for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
                mix[t] += xn[static_cast<size_t>(t) * hc * D + c * D + i] * sigm(a);
            }
        }
        if (lane == 0)
            for (int t = 0; t < T; ++t) mixed[static_cast<size_t>(t) * D + i] = mix[t] * (1.f / hc);
    }
}

// one warp per output i; each lane reads 8 bf16 at a time (R % 8 == 0, rows 16-byte aligned)
template <int NT>
__global__ void hc_up_mix_v_k(const uint16_t * __restrict__ W, const float * __restrict__ lo, int R, const float * __restrict__ xn,
                              float * __restrict__ mixed, int D, int hc) {
    extern __shared__ float ls[];   // [NT][R]
    for (int i = threadIdx.x; i < NT * R; i += blockDim.x) ls[i] = silu(lo[i] / hc);
    __syncthreads();
    const int lane = threadIdx.x & 31, nc = R / 8;
    for (int i = blockIdx.x * kWarps + (threadIdx.x >> 5); i < D; i += gridDim.x * kWarps) {
        float mix[NT] = {};
        for (int c = 0; c < hc; ++c) {
            const uint4 * row = reinterpret_cast<const uint4 *>(W + static_cast<size_t>(c * D + i) * R);
            float acc[NT] = {};
            for (int k = lane; k < nc; k += 32) {
                const uint4 v = row[k];
                const uint32_t u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float w0 = __uint_as_float(u[j] << 16), w1 = __uint_as_float(u[j] & 0xffff0000u);
#pragma unroll
                    for (int t = 0; t < NT; ++t) {
                        const float2 l2 = *reinterpret_cast<const float2 *>(ls + t * R + 8 * k + 2 * j);
                        acc[t] += w0 * l2.x + w1 * l2.y;
                    }
                }
            }
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                float a = acc[t];
                for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
                mix[t] += xn[static_cast<size_t>(t) * hc * D + c * D + i] * sigm(a);
            }
        }
        if (lane == 0)
#pragma unroll
            for (int t = 0; t < NT; ++t) mixed[static_cast<size_t>(t) * D + i] = mix[t] * (1.f / hc);
    }
}

// hc == 4: a warp per output i, 8 lanes per stream row (c = lane / 8), each lane R/64 chunks of 8 bf16 - every lane
// loads, all its loads at once; the 8-lane groups reduce their gate, then the 4 streams' gated values are summed
template <int NT, int CH>
__global__ void hc_up_mix_w4_k(const uint16_t * __restrict__ W, const float * __restrict__ lo, int R, const float * __restrict__ xn,
                               float * __restrict__ mixed, int D) {
    constexpr int hc = 4;
    extern __shared__ float ls[];   // [NT][R]
    for (int i = threadIdx.x; i < NT * R; i += blockDim.x) ls[i] = silu(lo[i] / hc);
    __syncthreads();
    const int lane = threadIdx.x & 31, c = lane >> 3, sub = lane & 7;
    for (int i = blockIdx.x * kWarps + (threadIdx.x >> 5); i < D; i += gridDim.x * kWarps) {
        const uint4 * row = reinterpret_cast<const uint4 *>(W + static_cast<size_t>(c * D + i) * R);
        uint4 v[CH];
#pragma unroll
        for (int j = 0; j < CH; ++j) v[j] = row[sub + 8 * j];
        float acc[NT] = {};
#pragma unroll
        for (int j = 0; j < CH; ++j) {
            const int k = 8 * (sub + 8 * j);
            const uint32_t u[4] = {v[j].x, v[j].y, v[j].z, v[j].w};
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                const float * l = ls + t * R + k;
                float sum = 0.f;
#pragma unroll
                for (int q = 0; q < 4; ++q) sum += __uint_as_float(u[q] << 16) * l[2 * q] + __uint_as_float(u[q] & 0xffff0000u) * l[2 * q + 1];
                acc[t] += sum;
            }
        }
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            float a = acc[t];
            for (int o = 4; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);   // within the 8-lane group: the gate
            float m = sub == 0 ? xn[static_cast<size_t>(t) * hc * D + c * D + i] * sigm(a) : 0.f;
            for (int o = 16; o >= 8; o >>= 1) m += __shfl_xor_sync(0xffffffffu, m, o);   // the 4 streams
            if (lane == 0) mixed[static_cast<size_t>(t) * D + i] = m * (1.f / hc);
        }
    }
}

__global__ void hc_combine_t_k(float * R, const float * h, const float * inj, int D, int hc, int T) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * hc * D) return;
    const int t = i / (hc * D), c = (i / D) % hc;
    R[i] += h[static_cast<size_t>(t) * D + i % D] * (2.f * sigm(inj[t * hc + c] / hc));
}

template <int F>
__global__ void embed_tokens_k(const uint8_t * W, size_t rb, int D, const int * tokens, float * out) {
    const uint8_t * row = W + static_cast<size_t>(tokens[blockIdx.x]) * rb;
    float * o = out + static_cast<size_t>(blockIdx.x) * D;
    for (int gg = threadIdx.x; gg < D / 8; gg += blockDim.x) {
        float v[8];
        dequant8<F>(row + static_cast<size_t>(gg / Grp<F>::groups) * Grp<F>::bytes, gg % Grp<F>::groups, v);
        for (int j = 0; j < 8; ++j) o[8 * gg + j] = v[j];
    }
}

__global__ void add_broadcast_streams_k(const float * h, const float * e, float * R, int D, int hc, int T) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * hc * D) return;
    R[i] = h[i] + e[(i / (hc * D)) * D + i % D];
}

__global__ void argmax_rows_k(const float * logits, int n, int * ids, float * probs) {
    const float * l = logits + static_cast<size_t>(blockIdx.x) * n;
    float best = -FLT_MAX;
    int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        if (l[i] > best) { best = l[i]; bi = i; }
    __shared__ float bv[32];
    __shared__ int bix[32];
    for (int o = 16; o > 0; o >>= 1) {
        const float v2 = __shfl_xor_sync(0xffffffffu, best, o);
        const int i2 = __shfl_xor_sync(0xffffffffu, bi, o);
        if (v2 > best || (v2 == best && i2 < bi)) { best = v2; bi = i2; }
    }
    if ((threadIdx.x & 31) == 0) { bv[threadIdx.x >> 5] = best; bix[threadIdx.x >> 5] = bi; }
    __syncthreads();
    __shared__ float top;
    if (threadIdx.x == 0) {
        for (int w = 1; w < static_cast<int>(blockDim.x >> 5); ++w)
            if (bv[w] > best || (bv[w] == best && bix[w] < bi)) { best = bv[w]; bi = bix[w]; }
        ids[blockIdx.x] = bi;
        top = best;
    }
    if (!probs) return;
    __syncthreads();
    float sum = 0.f;   // the winner's probability: 1 / sum exp(l - max)
    for (int i = threadIdx.x; i < n; i += blockDim.x) sum += expf(l[i] - top);
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) bv[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f;
        for (int w = 0; w < static_cast<int>(blockDim.x >> 5); ++w) t += bv[w];
        probs[blockIdx.x] = 1.f / t;
    }
}

// two passes: kArgParts blocks per row each reduce a slice to (max, its lowest index, sum exp(l - max)); then a merge
constexpr int kArgParts = 64;
struct ArgPart { float m; int i; float s; };
__device__ ArgPart g_arg[kMaxT * kArgParts];

__device__ __forceinline__ void arg_merge(float & m, int & i, float & s, float m2, int i2, float s2) {
    if (m2 > m || (m2 == m && i2 < i)) {
        s = s2 + (m == -FLT_MAX ? 0.f : s * expf(m - m2));
        m = m2;
        i = i2;
    } else if (m2 != -FLT_MAX) {
        s += s2 * expf(m2 - m);
    }
}

__global__ void argmax_part_k(const float * logits, int n) {
    const float * l = logits + static_cast<size_t>(blockIdx.y) * n;
    const int per = (n + kArgParts - 1) / kArgParts, lo = blockIdx.x * per, hi = min(n, lo + per);
    float m = -FLT_MAX, s = 0.f;
    int bi = 0x7fffffff;
    for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) {
        const float v = l[i];
        if (v > m) { s = s * expf(m - v) + 1.f; m = v; bi = i; }   // m == -FLT_MAX first: s * 0 + 1
        else s += expf(v - m);
    }
    for (int o = 16; o > 0; o >>= 1)
        arg_merge(m, bi, s, __shfl_xor_sync(0xffffffffu, m, o), __shfl_xor_sync(0xffffffffu, bi, o), __shfl_xor_sync(0xffffffffu, s, o));
    __shared__ ArgPart w[32];
    if ((threadIdx.x & 31) == 0) w[threadIdx.x >> 5] = {m, bi, s};
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int k = 1; k < static_cast<int>(blockDim.x >> 5); ++k) arg_merge(m, bi, s, w[k].m, w[k].i, w[k].s);
        g_arg[blockIdx.y * kArgParts + blockIdx.x] = {m, bi, s};
    }
}

__global__ void argmax_merge_k(int * ids, float * probs) {   // one warp per row
    const int lane = threadIdx.x;
    float m = -FLT_MAX, s = 0.f;
    int bi = 0x7fffffff;
    for (int k = lane; k < kArgParts; k += 32) {
        const ArgPart a = g_arg[blockIdx.x * kArgParts + k];
        arg_merge(m, bi, s, a.m, a.i, a.s);
    }
    for (int o = 16; o > 0; o >>= 1)
        arg_merge(m, bi, s, __shfl_xor_sync(0xffffffffu, m, o), __shfl_xor_sync(0xffffffffu, bi, o), __shfl_xor_sync(0xffffffffu, s, o));
    if (lane == 0) {
        ids[blockIdx.x] = bi;
        if (probs) probs[blockIdx.x] = 1.f / s;
    }
}

int grid_for(int rows) {
    const int want = (rows + kWarps - 1) / kWarps;
    return want < 82 * 8 ? want : 82 * 8;   // persistent-ish: x is staged in shared memory once per block
}

}  // namespace

void matvec_group(const MvGroup & g, const float * x, cudaStream_t s) {
    int rows = 0;
    for (int i = 0; i < g.n; ++i) rows += g.seg[i].M;
    if (g.K % 8 || g.K > 12288) throw std::runtime_error("matvec_group: K " + std::to_string(g.K));
    const size_t smem = static_cast<size_t>(g.K) * sizeof(float);
    if (rows < 1024)   // too few rows to fill the GPU a warp per row: a block per row
        mv_group_split_k<<<rows < 82 * 16 ? rows : 82 * 16, 32 * kWarps, smem, s>>>(g, x, rows);
    else
        mv_group_k<<<grid_for(rows), 32 * kWarps, smem, s>>>(g, x, rows);
}

static bool q8_enabled = std::getenv("BL_NO_Q8") == nullptr;
void set_matmul_q8(bool on) { q8_enabled = on; }

// a BF16 group with few rows and a long K (and no transform): the split-K kernel; false when it does not apply
static bool bf16_splitk(const MvGroup & g, const float * x, int T, int rows, cudaStream_t s) {
    if (g.xf != Xform::None || rows >= kBfMaxRows || g.K < 2048 || T > kMaxT) return false;
    for (int i = 0; i < g.n; ++i)
        if (g.seg[i].type != BF16) return false;
    const int tiles = (rows + kBfRows - 1) / kBfRows;
    // enough blocks to fill the GPU, slices of a multiple of 256 values, x slices within 48 KB at kMaxT tokens - the
    // split depends on the shape only, so a token's sum is the same in a window of any size
    int splits = 1;
    while (splits < kBfMaxSplit && (tiles * splits < 4 * 82 || static_cast<size_t>(kMaxT) * (g.K / splits) * 4 > 48 * 1024) &&
           g.K % (splits * 2) == 0 && (g.K / (splits * 2)) % 256 == 0)
        splits *= 2;
    const int kc = g.K / splits;
    if (kc % 256 || static_cast<size_t>(kMaxT) * kc * 4 > 48 * 1024) return false;
    const size_t sm = static_cast<size_t>(T) * kc * sizeof(float);
    const dim3 grid(tiles, splits);
    switch (T) {
        case 1: bf16_splitk_k<1><<<grid, 32 * kBfWarps, sm, s>>>(g, x, rows, kc); break;
        case 2: bf16_splitk_k<2><<<grid, 32 * kBfWarps, sm, s>>>(g, x, rows, kc); break;
        case 3: bf16_splitk_k<3><<<grid, 32 * kBfWarps, sm, s>>>(g, x, rows, kc); break;
        default: bf16_splitk_k<4><<<grid, 32 * kBfWarps, sm, s>>>(g, x, rows, kc); break;
    }
    return true;
}

void matmul_group(const MvGroup & g, const float * x, int T, cudaStream_t s) {
    int rows = 0;
    for (int i = 0; i < g.n; ++i) rows += g.seg[i].M;
    if (bf16_splitk(g, x, T, rows, s)) return;
    if (T == 1 && g.xf == Xform::None) return matvec_group(g, x, s);
    if (g.K % 8 || T < 1 || T > kMaxT) throw std::runtime_error("matmul_group: K " + std::to_string(g.K) + " T " + std::to_string(T));
    const bool split = rows < 1024;
    MvGroup gq = g;   // the quantized segments use dp4a on q8_1 activations when K allows
    gq.q8 = false;
    if (q8_enabled && g.K % 32 == 0 && g.K <= kMaxK8)
        for (int i = 0; i < g.n; ++i) gq.q8 |= has_dp4a(g.seg[i].type);
    if (gq.q8) {
        const int nb = T * g.K / 32;
        quantize_x_k<<<(nb + 7) / 8, 256, 0, s>>>(x, g.x2, g.xf, g.xs, nb);
    }
    auto launch = [&](auto nt) {
        constexpr int NT = decltype(nt)::value;
        // 16 lanes per row, 2 rows per warp: measured faster at K = 2560 and 6144 than a warp per row (and 8 lanes);
        // BL_MM_LPR=32 keeps a warp per row (testing)
        static const int force_lpr = std::getenv("BL_MM_LPR") ? std::atoi(std::getenv("BL_MM_LPR")) : 0;
        // (only on the q8 path: the fp32 one keeps a warp per row - the same sums as the one-token matvec)
        const bool half = gq.q8 && force_lpr != 32 && (g.K / 32) % 16 == 0;
        if (split) mm_group_split_k<NT><<<rows < 82 * 16 ? rows : 82 * 16, 32 * kWarps, 0, s>>>(gq, x, rows);
        else if (half) mm_group_lpr_k<NT, 16><<<grid_for((rows + 1) / 2), 32 * kWarps, 0, s>>>(gq, x, rows);
        else mm_group_k<NT><<<grid_for(rows), 32 * kWarps, 0, s>>>(gq, x, rows);
    };
    switch (T) {
        case 1: launch(std::integral_constant<int, 1>{}); break;
        case 2: launch(std::integral_constant<int, 2>{}); break;
        case 3: launch(std::integral_constant<int, 3>{}); break;
        default: launch(std::integral_constant<int, 4>{}); break;
    }
}

void hc_up_mix_t(const void * W, const float * lo, int R, const float * xn, float * mixed, int D, int hc, int T, cudaStream_t s) {
    if (hc == 4 && R == 320 && reinterpret_cast<uintptr_t>(W) % 16 == 0 && T >= 1 && T <= 4) {
        const auto * w = static_cast<const uint16_t *>(W);
        const size_t sm = static_cast<size_t>(T) * R * sizeof(float);
        switch (T) {
            case 1: hc_up_mix_w4_k<1, 5><<<grid_for(D), 32 * kWarps, sm, s>>>(w, lo, R, xn, mixed, D); return;
            case 2: hc_up_mix_w4_k<2, 5><<<grid_for(D), 32 * kWarps, sm, s>>>(w, lo, R, xn, mixed, D); return;
            case 3: hc_up_mix_w4_k<3, 5><<<grid_for(D), 32 * kWarps, sm, s>>>(w, lo, R, xn, mixed, D); return;
            default: hc_up_mix_w4_k<4, 5><<<grid_for(D), 32 * kWarps, sm, s>>>(w, lo, R, xn, mixed, D); return;
        }
    }
    if (R % 8 == 0 && reinterpret_cast<uintptr_t>(W) % 16 == 0) {
        const auto * w = static_cast<const uint16_t *>(W);
        const size_t sm = static_cast<size_t>(T) * R * sizeof(float);
        switch (T) {
            case 1: hc_up_mix_v_k<1><<<grid_for(D), 32 * kWarps, sm, s>>>(w, lo, R, xn, mixed, D, hc); return;
            case 2: hc_up_mix_v_k<2><<<grid_for(D), 32 * kWarps, sm, s>>>(w, lo, R, xn, mixed, D, hc); return;
            case 3: hc_up_mix_v_k<3><<<grid_for(D), 32 * kWarps, sm, s>>>(w, lo, R, xn, mixed, D, hc); return;
            case 4: hc_up_mix_v_k<4><<<grid_for(D), 32 * kWarps, sm, s>>>(w, lo, R, xn, mixed, D, hc); return;
        }
    }
    if (T == 1) return hc_up_mix(W, lo, R, xn, mixed, D, hc, s);
    hc_up_mix_t_k<<<grid_for(D), 32 * kWarps, static_cast<size_t>(T) * R * sizeof(float), s>>>(
        static_cast<const uint16_t *>(W), lo, R, xn, mixed, D, hc, T);
}

void hc_combine_t(float * R, const float * h, const float * inj, int D, int hc, int T, cudaStream_t s) {
    hc_combine_t_k<<<(T * hc * D + 255) / 256, 256, 0, s>>>(R, h, inj, D, hc, T);
}

bool embed_supported(uint32_t t) {
    switch (t) {
        case IQ3_S: case IQ4_NL: case IQ4_XS: case Q4_K: case Q5_K: case Q6_K: case Q8_0: case BF16: return true;
        default: return false;
    }
}

void embed_tokens(uint32_t t, const void * W, size_t rb, int D, const int * tokens, int T, float * out, cudaStream_t s) {
    const auto * w = static_cast<const uint8_t *>(W);
    switch (t) {
        case IQ3_S: embed_tokens_k<IQ3_S><<<T, 256, 0, s>>>(w, rb, D, tokens, out); break;
        case IQ4_NL: embed_tokens_k<IQ4_NL><<<T, 256, 0, s>>>(w, rb, D, tokens, out); break;
        case IQ4_XS: embed_tokens_k<IQ4_XS><<<T, 256, 0, s>>>(w, rb, D, tokens, out); break;
        case Q4_K:  embed_tokens_k<Q4_K><<<T, 256, 0, s>>>(w, rb, D, tokens, out); break;
        case Q5_K:  embed_tokens_k<Q5_K><<<T, 256, 0, s>>>(w, rb, D, tokens, out); break;
        case Q6_K:  embed_tokens_k<Q6_K><<<T, 256, 0, s>>>(w, rb, D, tokens, out); break;
        case Q8_0:  embed_tokens_k<Q8_0><<<T, 256, 0, s>>>(w, rb, D, tokens, out); break;
        case BF16:  embed_tokens_k<BF16><<<T, 256, 0, s>>>(w, rb, D, tokens, out); break;
        default: throw std::runtime_error("embed_tokens: format " + std::to_string(t));
    }
}

void argmax_rows(const float * logits, int n, int T, int * ids, cudaStream_t s, float * probs) {
    if (T <= kMaxT && n >= 32 * kArgParts) {
        argmax_part_k<<<dim3(kArgParts, T), 256, 0, s>>>(logits, n);
        argmax_merge_k<<<T, 32, 0, s>>>(ids, probs);
        return;
    }
    argmax_rows_k<<<T, 1024, 0, s>>>(logits, n, ids, probs);
}
void add_broadcast_streams(const float * h, const float * e, float * R, int D, int hc, int T, cudaStream_t s) {
    add_broadcast_streams_k<<<(T * hc * D + 255) / 256, 256, 0, s>>>(h, e, R, D, hc, T);
}

void hc_up_mix(const void * W, const float * lo, int R, const float * xn, float * mixed, int D, int hc, cudaStream_t s) {
    hc_up_mix_k<<<grid_for(D), 32 * kWarps, static_cast<size_t>(R) * sizeof(float), s>>>(
        static_cast<const uint16_t *>(W), lo, R, xn, mixed, D, hc);
}

void gdn_pre(float * st, const float * qkv, const float * w, float * out, int Hk, int Hv, int S, float eps, const float * b_raw,
             const float * a_raw, const float * dt, const float * A, float * beta, float * g, cudaStream_t s) {
    gdn_pre_k<<<2 * Hk + Hv + 1, S, 0, s>>>(st, qkv, w, out, Hk, Hv, S, eps, b_raw, a_raw, dt, A, beta, g);
}

void gdn_post(float * o, const float * w, const float * z, int Hv, int S, float eps, cudaStream_t s) {
    gdn_post_k<<<Hv, S, 0, s>>>(o, w, z, S, eps);
}

void qsa_pre(const float * qf, float * qo, int H, const float * k, const float * v, int Hkv, int Dh, const float * qn,
             const float * kn, float eps, int n_rot, float base, const TokenState * ts, float * kc, float * vc, cudaStream_t s) {
    qsa_pre_k<<<H + 2 * Hkv, Dh, static_cast<size_t>(Dh) * sizeof(float), s>>>(qf, qo, H, k, v, Hkv, Dh, qn, kn, eps, n_rot,
                                                                               base, ts, kc, vc);
}

void attention_gated(const float * q, const float * kc, const float * vc, const float * gate, int gstride, float * out,
                     int heads, int kv_heads, int D, const TokenState * ts, int max_kv, float scale, cudaStream_t s) {
    attention_gated_k<<<heads, 256, static_cast<size_t>(max_kv) * sizeof(float), s>>>(q, kc, vc, gate, gstride, out, heads,
                                                                                       kv_heads, D, ts, scale);
}

void ffn_combine(const float * eo, const float * w, int n, const float * sh, const float * sg, float * h, int D, cudaStream_t s) {
    ffn_combine_k<<<(D + 255) / 256, 256, 0, s>>>(eo, w, n, sh, sg, h, D);
}

namespace {
template <int T>
__global__ void embed_k(const uint8_t * W, size_t rb, int D, const TokenState * ts, float * out) {
    const uint8_t * row = W + static_cast<size_t>(ts->token) * rb;
    for (int gg = threadIdx.x; gg < D / 8; gg += blockDim.x) {
        float v[8];
        dequant8<T>(row + static_cast<size_t>(gg / Grp<T>::groups) * Grp<T>::bytes, gg % Grp<T>::groups, v);
        for (int j = 0; j < 8; ++j) out[8 * gg + j] = v[j];
    }
}
}  // namespace

void embed_token(uint32_t t, const void * W, size_t rb, int D, const TokenState * ts, float * out, cudaStream_t s) {
    const auto * w = static_cast<const uint8_t *>(W);
    switch (t) {
        case IQ3_S:  embed_k<IQ3_S><<<1, 256, 0, s>>>(w, rb, D, ts, out); break;
        case IQ4_XS: embed_k<IQ4_XS><<<1, 256, 0, s>>>(w, rb, D, ts, out); break;
        case Q4_K:   embed_k<Q4_K><<<1, 256, 0, s>>>(w, rb, D, ts, out); break;
        case Q5_K:   embed_k<Q5_K><<<1, 256, 0, s>>>(w, rb, D, ts, out); break;
        case Q6_K:   embed_k<Q6_K><<<1, 256, 0, s>>>(w, rb, D, ts, out); break;
        case BF16:   embed_k<BF16><<<1, 256, 0, s>>>(w, rb, D, ts, out); break;
        default: throw std::runtime_error("embed_token: format " + std::to_string(t));
    }
}

}  // namespace bl::cuda
