// Phase-1 forward kernels (one token, fp32): plain and checkable against llama.cpp, not yet fast.
#include <cfloat>
#include <stdexcept>
#include <string>

#include "bl/cuda_ops.h"

namespace bl::cuda {

namespace {

__device__ __forceinline__ float sigmoidf_(float x) { return 1.f / (1.f + expf(-x)); }
__device__ __forceinline__ float siluf_(float x) { return x / (1.f + expf(-x)); }

// block-wide sum; blockDim.x a multiple of 32, at most 1024
__device__ float block_sum(float v) {
    __shared__ float part[32];
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) part[w] = v;
    __syncthreads();
    const int nw = blockDim.x >> 5;
    v = l < nw ? part[l] : 0.f;
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
    const int nw = blockDim.x >> 5;
    v = l < nw ? part[l] : -FLT_MAX;
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

inline int blocks(int n, int t = 256) { return (n + t - 1) / t; }

__global__ void rmsnorm_rows_k(const float * x, float * y, int D, const float * w, int w_rows, float eps, float post) {
    const float * xr = x + static_cast<size_t>(blockIdx.x) * D;
    float * yr = y + static_cast<size_t>(blockIdx.x) * D;
    float ss = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss);
    const float scale = 1.f / sqrtf(ss / D + eps);
    const float * wr = w ? (w_rows > 1 ? w + static_cast<size_t>(blockIdx.x % w_rows) * D : w) : nullptr;
    for (int i = threadIdx.x; i < D; i += blockDim.x) yr[i] = xr[i] * scale * (wr ? wr[i] : 1.f) * post;
}

__global__ void silu_scale_k(float * x, int n, float scale) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = siluf_(x[i] * scale);
}
__global__ void sigmoid_mul_k(float * x, const float * g, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= sigmoidf_(g[i]);
}
__global__ void sigmoid_scale_k(float * x, const float * sc, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= sigmoidf_(*sc);
}
__global__ void swiglu_k(const float * g, const float * u, float * o, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = siluf_(g[i]) * u[i];
}
__global__ void axpy_k(float * y, const float * x, float a, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a * x[i];
}
__global__ void add_k(float * y, const float * x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += x[i];
}
__global__ void repeat_k(const float * x, float * y, int D, int times) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < D * times) y[i] = x[i % D];
}

// mixed = (1/hc) * sum_c xn[c] * sigmoid(gate[c])
__global__ void hc_gate_mean_k(const float * xn, const float * gate, float * mixed, int D, int hc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= D) return;
    float acc = 0.f;
    for (int c = 0; c < hc; ++c) acc += xn[c * D + i] * sigmoidf_(gate[c * D + i]);
    mixed[i] = acc * (1.f / hc);
}

// R[c] += h * 2*sigmoid(inj[c] / hc)
__global__ void hc_combine_k(float * R, const float * h, const float * inj, int D, int hc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= D * hc) return;
    const int c = i / D;
    R[i] += h[i % D] * (2.f * sigmoidf_(inj[c] / hc));
}

// ggml_ssm_conv over [history(3) | x] then silu; the newest three inputs become the history
__global__ void gdn_conv_k(float * state, const float * x, const float * w, float * out, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float * st = state + 3 * c;
    const float * wc = w + 4 * c;
    const float xc = x[c];
    const float acc = st[0] * wc[0] + st[1] * wc[1] + st[2] * wc[2] + xc * wc[3];
    out[c] = siluf_(acc);
    st[0] = st[1]; st[1] = st[2]; st[2] = xc;
}

__global__ void gdn_gates_k(const float * b, const float * a, const float * dt, const float * A, float * beta, float * g, int H) {
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= H) return;
    beta[h] = sigmoidf_(b[h]);
    const float z = a[h] + dt[h];
    const float sp = z > 20.f ? z : logf(1.f + expf(z));   // ggml op_softplus
    g[h] = sp * A[h];
}

// thread j owns column j of one head's S (S[i][j], i = key index), held in registers: the state is read and written
// once per step. A head's columns are independent, so they spread over kSplit blocks. ggml gated_delta_net
constexpr int kGdnS = 128, kGdnSplit = 4;
__global__ void __launch_bounds__(kGdnS / kGdnSplit) gdn_step_k(float * state, const float * q, const float * k,
                                                                 const float * v, const float * g, const float * beta,
                                                                 float * out, int H_k) {
    constexpr int S = kGdnS;
    const int h = blockIdx.x / kGdnSplit, j = (blockIdx.x % kGdnSplit) * (S / kGdnSplit) + threadIdx.x;
    float * st = state + static_cast<size_t>(h) * S * S;
    __shared__ float qs[S], ks[S];
    for (int i = threadIdx.x; i < S; i += blockDim.x) {
        qs[i] = q[(h % H_k) * S + i];
        ks[i] = k[(h % H_k) * S + i];
    }
    __syncthreads();
    const float decay = expf(g[h]);
    float col[S];
    float sk = 0.f;
#pragma unroll
    for (int i = 0; i < S; ++i) {
        col[i] = st[i * S + j] * decay;
        sk += col[i] * ks[i];
    }
    const float d = (v[h * S + j] - sk) * beta[h];
    float o = 0.f;
#pragma unroll
    for (int i = 0; i < S; ++i) {
        col[i] += ks[i] * d;
        o += col[i] * qs[i];
        st[i * S + j] = col[i];
    }
    out[h * S + j] = o * (1.f / sqrtf(static_cast<float>(S)));
}

__global__ void split_q_gate_k(const float * qf, float * q, float * gate, int heads, int D) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= heads * D) return;
    const int h = i / D, d = i % D;
    q[i]    = qf[h * 2 * D + d];
    gate[i] = qf[h * 2 * D + D + d];
}

// NEOX pairing (i, i + n_rot/2) on the first n_rot dims; theta_i = pos * base^(-2i/n_rot)
__global__ void rope_neox_k(float * x, int D, int n_rot, int pos, float base) {
    const int h = blockIdx.x, i = threadIdx.x;
    const int half = n_rot / 2;
    if (i >= half) return;
    const float theta = pos * powf(base, -2.f * i / n_rot);
    float sn, cs;
    sincosf(theta, &sn, &cs);
    float * xh = x + h * D;
    const float x0 = xh[i], x1 = xh[i + half];
    xh[i]        = x0 * cs - x1 * sn;
    xh[i + half] = x0 * sn + x1 * cs;
}

// one block per query head: scores over n_kv positions in shared memory, softmax, weighted V
__global__ void attention_k(const float * q, const float * kc, const float * vc, float * out, int heads, int kv_heads,
                            int D, int n_kv, float scale) {
    extern __shared__ float sc[];
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
        out[h * D + d] = acc / sum;
    }
}

// single block: softmax probabilities, then k rounds of argmax (ties: lowest id)
__global__ void router_topk_k(const float * logits, int n, int k, int * ids, float * w) {
    extern __shared__ float pr[];
    float m = -FLT_MAX;
    for (int i = threadIdx.x; i < n; i += blockDim.x) m = fmaxf(m, logits[i]);
    m = block_max(m);
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        pr[i] = expf(logits[i] - m);
        s += pr[i];
    }
    s = block_sum(s);
    for (int i = threadIdx.x; i < n; i += blockDim.x) pr[i] /= s;
    __syncthreads();
    __shared__ float bv[32];
    __shared__ int   bi[32];
    __shared__ float wsum;
    if (threadIdx.x == 0) wsum = 0.f;
    for (int r = 0; r < k; ++r) {
        float v = -FLT_MAX;
        int id = n;
        for (int i = threadIdx.x; i < n; i += blockDim.x)
            if (pr[i] > v || (pr[i] == v && i < id)) { v = pr[i]; id = i; }
        for (int o = 16; o > 0; o >>= 1) {
            const float v2 = __shfl_xor_sync(0xffffffffu, v, o);
            const int   i2 = __shfl_xor_sync(0xffffffffu, id, o);
            if (v2 > v || (v2 == v && i2 < id)) { v = v2; id = i2; }
        }
        if ((threadIdx.x & 31) == 0) { bv[threadIdx.x >> 5] = v; bi[threadIdx.x >> 5] = id; }
        __syncthreads();
        if (threadIdx.x == 0) {
            float bvv = bv[0];
            int bii = bi[0];
            for (int t = 1; t < static_cast<int>(blockDim.x >> 5); ++t)
                if (bv[t] > bvv || (bv[t] == bvv && bi[t] < bii)) { bvv = bv[t]; bii = bi[t]; }
            ids[r] = bii;
            w[r] = bvv;
            wsum += bvv;
            pr[bii] = -FLT_MAX;
        }
        __syncthreads();
    }
    if (threadIdx.x < k) w[threadIdx.x] /= fmaxf(wsum, 6.103515625e-5f);
}

// up to two routers in one launch (blockIdx.x picks one); one thread per expert (n <= blockDim.x = 512).
// Each round: warp argmax by shuffles, then warp 0 merges the 16 warp winners. Same result as router_topk_k.
__global__ void router_topk2_k(const float * l0, int * ids0, float * w0, const float * l1, int * ids1, float * w1, int n, int k) {
    const float * logits = blockIdx.x ? l1 : l0;
    int * ids = blockIdx.x ? ids1 : ids0;
    float * w = blockIdx.x ? w1 : w0;
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

// s[c] = sum(key_n[c] * query_n[c]) / sqrt(D);  gate = sigmoid(sign(s) * sqrt(max(|s|, 1e-6)))
__global__ void ple_gate_k(const float * kn, const float * qn, float * gate, int D) {
    const int c = blockIdx.x;
    float acc = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) acc += kn[c * D + i] * qn[c * D + i];
    acc = block_sum(acc);
    if (threadIdx.x == 0) {
        const float s = acc * (1.f / sqrtf(static_cast<float>(D)));
        const float mag = sqrtf(fmaxf(fabsf(s), 1e-6f));
        const float sg = s > 0.f ? 1.f : (s < 0.f ? -1.f : 0.f);
        gate[c] = sigmoidf_(sg * mag);
    }
}

__global__ void ple_gated_value_k(const float * value, const float * gate, float * gated, int D, int hc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < D * hc) gated[i] = value[i % D] * gate[i / D];
}

__global__ void ple_conv_k(float * hist, const float * x, const float * w, float * out, int C, int kern, int dil) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int n_hist = (kern - 1) * dil;
    float acc = 0.f;
    for (int k = 0; k < kern; ++k) {
        const int back = (kern - 1 - k) * dil;   // positions before the current token
        const float v = back == 0 ? x[c] : hist[static_cast<size_t>(n_hist - back) * C + c];
        acc += w[c * kern + k] * v;
    }
    out[c] = siluf_(acc);
    for (int r = 0; r + 1 < n_hist; ++r) hist[static_cast<size_t>(r) * C + c] = hist[static_cast<size_t>(r + 1) * C + c];
    hist[static_cast<size_t>(n_hist - 1) * C + c] = x[c];
}

}  // namespace

void rmsnorm_rows(const float * x, float * y, int rows, int D, const float * w, int w_rows, float eps, float post, cudaStream_t s) {
    rmsnorm_rows_k<<<rows, 256, 0, s>>>(x, y, D, w, w_rows, eps, post);
}
void silu_scale(float * x, int n, float scale, cudaStream_t s) { silu_scale_k<<<blocks(n), 256, 0, s>>>(x, n, scale); }
void sigmoid_mul(float * x, const float * g, int n, cudaStream_t s) { sigmoid_mul_k<<<blocks(n), 256, 0, s>>>(x, g, n); }
void sigmoid_scale(float * x, const float * sc, int n, cudaStream_t s) { sigmoid_scale_k<<<blocks(n), 256, 0, s>>>(x, sc, n); }
void swiglu(const float * g, const float * u, float * o, int n, cudaStream_t s) { swiglu_k<<<blocks(n), 256, 0, s>>>(g, u, o, n); }
void axpy(float * y, const float * x, float a, int n, cudaStream_t s) { axpy_k<<<blocks(n), 256, 0, s>>>(y, x, a, n); }
void add(float * y, const float * x, int n, cudaStream_t s) { add_k<<<blocks(n), 256, 0, s>>>(y, x, n); }
void repeat(const float * x, float * y, int D, int t, cudaStream_t s) { repeat_k<<<blocks(D * t), 256, 0, s>>>(x, y, D, t); }

void hc_gate_mean(const float * xn, const float * gate, float * mixed, int D, int hc, cudaStream_t s) {
    hc_gate_mean_k<<<blocks(D), 256, 0, s>>>(xn, gate, mixed, D, hc);
}
void hc_combine(float * R, const float * h, const float * inj, int D, int hc, cudaStream_t s) {
    hc_combine_k<<<blocks(D * hc), 256, 0, s>>>(R, h, inj, D, hc);
}

void gdn_conv(float * state, const float * x, const float * w, float * out, int C, cudaStream_t s) {
    gdn_conv_k<<<blocks(C), 256, 0, s>>>(state, x, w, out, C);
}
void gdn_gates(const float * b, const float * a, const float * dt, const float * A, float * beta, float * g, int H, cudaStream_t s) {
    gdn_gates_k<<<blocks(H, 64), 64, 0, s>>>(b, a, dt, A, beta, g, H);
}
void gdn_step(float * state, const float * q, const float * k, const float * v, const float * g, const float * beta,
              float * out, int H_v, int H_k, int S, cudaStream_t s) {
    if (S != kGdnS) throw std::runtime_error("gdn_step: state size " + std::to_string(S));
    gdn_step_k<<<H_v * kGdnSplit, kGdnS / kGdnSplit, 0, s>>>(state, q, k, v, g, beta, out, H_k);
}

void split_q_gate(const float * qf, float * q, float * gate, int heads, int D, cudaStream_t s) {
    split_q_gate_k<<<blocks(heads * D), 256, 0, s>>>(qf, q, gate, heads, D);
}
void rope_neox(float * x, int heads, int D, int n_rot, int pos, float base, cudaStream_t s) {
    rope_neox_k<<<heads, n_rot / 2, 0, s>>>(x, D, n_rot, pos, base);
}
void attention(const float * q, const float * kc, const float * vc, float * out, int heads, int kv_heads, int D, int n_kv,
               float scale, cudaStream_t s) {
    attention_k<<<heads, 256, static_cast<size_t>(n_kv) * sizeof(float), s>>>(q, kc, vc, out, heads, kv_heads, D, n_kv, scale);
}

void router_topk(const float * logits, int n, int k, int * ids, float * w, cudaStream_t s) {
    router_topk_k<<<1, 512, static_cast<size_t>(n) * sizeof(float), s>>>(logits, n, k, ids, w);
}

void router_topk2(const float * l0, int * ids0, float * w0, const float * l1, int * ids1, float * w1, int n, int k, cudaStream_t s) {
    if (n > 512) throw std::runtime_error("router_topk2: more than 512 experts");
    router_topk2_k<<<l1 ? 2 : 1, 512, 0, s>>>(l0, ids0, w0, l1, ids1, w1, n, k);
}

void ple_gate(const float * kn, const float * qn, float * gate, int D, int hc, cudaStream_t s) {
    ple_gate_k<<<hc, 256, 0, s>>>(kn, qn, gate, D);
}
void ple_gated_value(const float * value, const float * gate, float * gated, int D, int hc, cudaStream_t s) {
    ple_gated_value_k<<<blocks(D * hc), 256, 0, s>>>(value, gate, gated, D, hc);
}
void ple_conv(float * hist, const float * x, const float * w, float * out, int C, int kern, int dil, cudaStream_t s) {
    ple_conv_k<<<blocks(C), 256, 0, s>>>(hist, x, w, out, C, kern, dil);
}

}  // namespace bl::cuda
