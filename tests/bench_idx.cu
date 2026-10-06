// Time the QSA indexer's selection (idx_select: score every complete block, then pick the top 512) for T queries
// ending at several context positions, and estimate a whole prefill's indexer time from them.
//   bench_idx [T] [ctx]      (T = 256: a prefill sub-chunk; T = 4: a decode window)
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "bl/cuda_window.h"

int main(int argc, char ** argv) {
    const int T = argc > 1 ? std::atoi(argv[1]) : 256;
    const int ctx = argc > 2 ? std::atoi(argv[2]) : 131072;
    constexpr int Hq = 4, Di = 128, K = 512, kStride = 2056, kSc = 256, kLayers = 12;
    const int max_blocks = ctx / 4 + 1;

    // queries and block keys like the model's: rms-normed rows (unit scale), random directions
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> hq(static_cast<size_t>(T) * Hq * Di), hk(static_cast<size_t>(max_blocks) * Di);
    for (auto & v : hq) v = nd(rng);
    for (auto & v : hk) v = nd(rng);
    float *q, *kblk, *score;
    int *list, *cnt;
    bl::cuda::TokenState * ts;
    cudaMalloc(&q, hq.size() * 4);
    cudaMalloc(&kblk, hk.size() * 4);
    cudaMalloc(&score, static_cast<size_t>(T) * max_blocks * 4);
    cudaMalloc(&list, static_cast<size_t>(T) * kStride * 4);
    cudaMalloc(&cnt, T * 4);
    cudaMalloc(&ts, sizeof(bl::cuda::TokenState));
    cudaMemcpy(q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(kblk, hk.data(), hk.size() * 4, cudaMemcpyHostToDevice);
    cudaStream_t s;
    cudaStreamCreate(&s);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    auto time_at = [&](int pos) {   // ms per call, queries at pos .. pos + T - 1
        bl::cuda::TokenState h{};
        h.pos = pos;
        cudaMemcpy(ts, &h, sizeof h, cudaMemcpyHostToDevice);
        bl::cuda::idx_select(q, kblk, ts, Hq, Di, K, score, max_blocks, list, kStride, cnt, T, s);   // warm-up
        const int reps = 10;
        cudaEventRecord(e0, s);
        for (int r = 0; r < reps; ++r)
            bl::cuda::idx_select(q, kblk, ts, Hq, Di, K, score, max_blocks, list, kStride, cnt, T, s);
        cudaEventRecord(e1, s);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        return ms / reps;
    };

    // the new scoring kernel against the first one: the same scores (bit for bit) and the same selection
    auto check_at = [&](int pos) {
        bl::cuda::TokenState h{};
        h.pos = pos;
        cudaMemcpy(ts, &h, sizeof h, cudaMemcpyHostToDevice);
        const size_t ns = static_cast<size_t>(T) * max_blocks, nl = static_cast<size_t>(T) * kStride;
        std::vector<float> s0(ns), s1(ns);
        std::vector<int> l0(nl), l1(nl), c0(T), c1(T);
        bl::cuda::idx_select_ref(q, kblk, ts, Hq, Di, K, score, max_blocks, list, kStride, cnt, T, s);
        cudaMemcpy(s0.data(), score, ns * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(l0.data(), list, nl * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(c0.data(), cnt, T * 4, cudaMemcpyDeviceToHost);
        bl::cuda::idx_select(q, kblk, ts, Hq, Di, K, score, max_blocks, list, kStride, cnt, T, s);
        cudaMemcpy(s1.data(), score, ns * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(l1.data(), list, nl * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(c1.data(), cnt, T * 4, cudaMemcpyDeviceToHost);
        long bad_s = 0, bad_l = 0;
        for (int t = 0; t < T; ++t) {
            const int nb = (pos + t + 1) / 4;
            if (nb > K)
                for (int b = 0; b < nb; ++b) bad_s += s0[static_cast<size_t>(t) * max_blocks + b] != s1[static_cast<size_t>(t) * max_blocks + b];
            if (c0[t] != c1[t]) { ++bad_l; continue; }
            for (int j = 0; j < c0[t]; ++j) bad_l += l0[static_cast<size_t>(t) * kStride + j] != l1[static_cast<size_t>(t) * kStride + j];
        }
        std::printf("  check at %6d: %ld scores differ, %ld selection entries differ\n", pos, bad_s, bad_l);
        return bad_s == 0 && bad_l == 0;
    };
    bool ok = true;
    for (int pos : {100, 2000, 2100, 9000, ctx - T}) ok = check_at(pos) && ok;
    if (!ok) { std::printf("MISMATCH\n"); return 1; }

    std::printf("T = %d queries per call, Hq %d x Di %d, top %d blocks\n", T, Hq, Di, K);
    for (int pos : {8192, 32768, 65536, ctx - T})
        if (pos + T <= ctx) std::printf("  at %6d: %8.3f ms\n", pos, time_at(pos));
    if (T == kSc) {   // a prefill of ctx tokens: ctx / kSc sub-chunks per QSA layer, sampled every 16
        double total = 0;
        const int step = 16 * kSc;
        for (int pos = 0; pos + kSc <= ctx; pos += step) total += time_at(pos) * 16;
        std::printf("prefill of %d tokens: ~%.1f s of indexer selection (%d QSA layers)\n", ctx, total * kLayers / 1000, kLayers);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { std::printf("CUDA error: %s\n", cudaGetErrorString(err)); return 1; }
    return 0;
}
