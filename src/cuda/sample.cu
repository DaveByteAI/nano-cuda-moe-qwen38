#include "bl/cuda_sample.h"

#include <cfloat>
#include <stdexcept>

namespace bl::cuda {

namespace {

constexpr int kThreads = 1024;

// an order-preserving map of a float's bits onto unsigned integers
__device__ __forceinline__ unsigned okey(float f) {
    const unsigned u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__global__ void sample_rows_k(const float * logits, int V, SampleParams p, const int * draft, const float * uni, int * out_tok,
                             int * out_acc) {
    const int row = blockIdx.x, tid = threadIdx.x;
    const float * l = logits + static_cast<size_t>(row) * V;
    const int K = p.top_k <= 0 || p.top_k > kSampleMaxK ? kSampleMaxK : p.top_k;
    __shared__ int hist[256];
    __shared__ unsigned s_prefix;
    __shared__ int s_rem;
    // 1. the K-th largest key: radix select, 8 bits a pass
    unsigned prefix = 0;
    int rem = K;
    for (int shift = 24; shift >= 0; shift -= 8) {
        const unsigned mask_hi = shift == 24 ? 0u : ~((1u << (shift + 8)) - 1);
        for (int i = tid; i < 256; i += blockDim.x) hist[i] = 0;
        __syncthreads();
        for (int i = tid; i < V; i += blockDim.x) {
            const unsigned k = okey(l[i]);
            if ((k & mask_hi) == prefix) atomicAdd(&hist[(k >> shift) & 255], 1);
        }
        __syncthreads();
        if (tid == 0) {
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
    const unsigned thr = prefix;
    // 2. the candidates: every key above the threshold, then ties at it (the lowest ids, after the sort)
    __shared__ float c_l[2 * kSampleMaxK];
    __shared__ int c_i[2 * kSampleMaxK];
    __shared__ int n_gt, n_eq;
    if (tid == 0) { n_gt = 0; n_eq = 0; }
    __syncthreads();
    for (int i = tid; i < V; i += blockDim.x) {
        const unsigned k = okey(l[i]);
        if (k > thr) {
            const int s = atomicAdd(&n_gt, 1);
            if (s < kSampleMaxK) { c_l[s] = l[i]; c_i[s] = i; }
        } else if (k == thr) {
            const int s = atomicAdd(&n_eq, 1);
            if (s < kSampleMaxK) { c_l[kSampleMaxK + s] = l[i]; c_i[kSampleMaxK + s] = i; }
        }
    }
    __syncthreads();
    // 3. thread 0: order (logit desc, id asc), softmax, nucleus, then the draw / the rejection step
    if (tid == 0) {
        const int ng = min(n_gt, K), ne = min(n_eq, kSampleMaxK);
        // ties: the lowest ids first
        for (int a = 1; a < ne; ++a) {
            const int id = c_i[kSampleMaxK + a];
            int b = a - 1;
            while (b >= 0 && c_i[kSampleMaxK + b] > id) { c_i[kSampleMaxK + b + 1] = c_i[kSampleMaxK + b]; --b; }
            c_i[kSampleMaxK + b + 1] = id;
        }
        int n = ng;
        for (int a = 0; a < ne && n < K; ++a, ++n) { c_l[n] = l[c_i[kSampleMaxK + a]]; c_i[n] = c_i[kSampleMaxK + a]; }
        for (int a = 1; a < n; ++a) {   // insertion sort: at most 256 values
            const float v = c_l[a];
            const int id = c_i[a];
            int b = a - 1;
            while (b >= 0 && (c_l[b] < v || (c_l[b] == v && c_i[b] > id))) { c_l[b + 1] = c_l[b]; c_i[b + 1] = c_i[b]; --b; }
            c_l[b + 1] = v;
            c_i[b + 1] = id;
        }
        const float inv_t = 1.f / p.temperature, m = c_l[0];
        float z = 0.f;
        for (int a = 0; a < n; ++a) { c_l[a] = expf((c_l[a] - m) * inv_t); z += c_l[a]; }
        // nucleus: the smallest prefix whose mass reaches top_p
        int keep = n;
        float zk = z;
        if (p.top_p < 1.f) {
            float cum = 0.f;
            for (int a = 0; a < n; ++a) {
                cum += c_l[a];
                if (cum >= p.top_p * z) { keep = a + 1; zk = cum; break; }
            }
        }
        const int d = draft ? draft[row] : -1;
        const float u1 = uni[2 * row], u2 = uni[2 * row + 1];
        int jd = -1;
        if (d >= 0)
            for (int a = 0; a < keep; ++a)
                if (c_i[a] == d) { jd = a; break; }
        if (d >= 0) {
            const float pd = jd >= 0 ? c_l[jd] / zk : 0.f;
            if (u1 < pd) { out_tok[row] = d; out_acc[row] = 1; return; }
        }
        // a draw from the kept candidates, without the rejected draft
        const float total = zk - (jd >= 0 ? c_l[jd] : 0.f);
        float target = u2 * total, cum = 0.f;
        int pick = -1;
        for (int a = 0; a < keep; ++a) {
            if (a == jd) continue;
            cum += c_l[a];
            pick = a;
            if (cum > target) break;
        }
        out_tok[row] = pick >= 0 ? c_i[pick] : c_i[0];
        out_acc[row] = 0;
    }
}

}  // namespace

void sample_rows(const float * logits, int V, int T, const SampleParams & p, const int * draft, const float * uni, int * out_tok,
                 int * out_acc, cudaStream_t s) {
    if (!(p.temperature > 0.f)) throw std::runtime_error("sample_rows: temperature must be > 0 (greedy is the argmax path)");
    sample_rows_k<<<T, kThreads, 0, s>>>(logits, V, p, draft, uni, out_tok, out_acc);
}

}  // namespace bl::cuda
