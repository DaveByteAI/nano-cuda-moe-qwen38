// The GPU sampler against the exact distribution (top_k, temperature, top_p) computed on the host:
//   - plain draws (no draft) follow it;
//   - with a deterministic draft d, "accept d with p(d), else draw from p without d" follows it too (the property
//     speculative sampling relies on), for a likely and an unlikely d.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "bl/cuda_sample.h"

int main() {
    const int V = 2000, rows = 4096, rounds = 50;   // 204800 draws per case
    std::mt19937 rng(3);
    std::normal_distribution<float> N(0.f, 2.f);
    std::vector<float> logit(V);
    for (auto & v : logit) v = N(rng);
    logit[17] = logit[18] = 4.5f;   // a tie at the top
    bl::cuda::SampleParams p;
    p.temperature = 0.8f;
    p.top_k = 50;
    p.top_p = 0.9f;
    // the exact distribution
    std::vector<int> ord(V);
    std::iota(ord.begin(), ord.end(), 0);
    std::sort(ord.begin(), ord.end(), [&](int a, int b) { return logit[a] != logit[b] ? logit[a] > logit[b] : a < b; });
    std::vector<double> prob(V, 0.0);
    double z = 0;
    for (int i = 0; i < p.top_k; ++i) z += std::exp((logit[ord[i]] - logit[ord[0]]) / p.temperature);
    double cum = 0, zk = 0;
    int keep = p.top_k;
    for (int i = 0; i < p.top_k; ++i) {
        cum += std::exp((logit[ord[i]] - logit[ord[0]]) / p.temperature);
        if (cum >= p.top_p * z) { keep = i + 1; zk = cum; break; }
    }
    for (int i = 0; i < keep; ++i) prob[ord[i]] = std::exp((logit[ord[i]] - logit[ord[0]]) / p.temperature) / zk;

    float *dl, *du;
    int *dd, *dt, *da;
    cudaMalloc(&dl, sizeof(float) * V * rows);
    cudaMalloc(&du, sizeof(float) * 2 * rows);
    cudaMalloc(&dd, sizeof(int) * rows);
    cudaMalloc(&dt, sizeof(int) * rows);
    cudaMalloc(&da, sizeof(int) * rows);
    for (int r = 0; r < rows; ++r) cudaMemcpy(dl + static_cast<size_t>(r) * V, logit.data(), sizeof(float) * V, cudaMemcpyHostToDevice);
    std::uniform_real_distribution<float> U(0.f, 1.f);
    int fails = 0;
    for (int draft : {-1, ord[0], ord[keep - 1], ord[keep + 3]}) {   // none, the likeliest, the least likely kept, one cut
        std::vector<double> cnt(V, 0.0);
        std::vector<int> d(rows, draft), tok(rows), acc(rows);
        cudaMemcpy(dd, d.data(), sizeof(int) * rows, cudaMemcpyHostToDevice);
        long accepted = 0;
        for (int k = 0; k < rounds; ++k) {
            std::vector<float> u(2 * rows);
            for (auto & x : u) x = U(rng);
            cudaMemcpy(du, u.data(), sizeof(float) * 2 * rows, cudaMemcpyHostToDevice);
            bl::cuda::sample_rows(dl, V, rows, p, dd, du, dt, da, nullptr);
            cudaMemcpy(tok.data(), dt, sizeof(int) * rows, cudaMemcpyDeviceToHost);
            cudaMemcpy(acc.data(), da, sizeof(int) * rows, cudaMemcpyDeviceToHost);
            for (int r = 0; r < rows; ++r) { cnt[tok[r]] += 1; accepted += acc[r]; }
        }
        const double n = static_cast<double>(rows) * rounds;
        double max_dev = 0, outside = 0;   // the largest |frequency - p| in units of its standard error; mass outside p
        for (int i = 0; i < V; ++i) {
            const double f = cnt[i] / n;
            if (prob[i] == 0) { outside += f; continue; }
            max_dev = std::max(max_dev, std::fabs(f - prob[i]) / std::sqrt(prob[i] * (1 - prob[i]) / n));
        }
        const bool ok = max_dev < 5.0 && outside == 0;
        std::printf("draft %5d (p %.4f): accepted %.4f, max deviation %.2f sigma over %d kept tokens, mass outside %.1e  %s\n",
                    draft, draft >= 0 ? prob[draft] : 0.0, accepted / n, max_dev, keep, outside, ok ? "ok" : "FAIL");
        fails += !ok;
    }
    std::printf("%s\n", fails ? "SAMPLER MISMATCH" : "the sampler follows the target distribution, with or without drafts");
    return fails ? 1 : 0;
}
