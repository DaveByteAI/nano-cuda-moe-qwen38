// The q8_1/dp4a multi-token matmul against the fp32 one, per weight format, on real rows of the model.
//   test_mmq SHARD1.gguf
#include <cmath>
#include <cstdio>
#include <map>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "bl/cuda_fused.h"
#include "bl/gguf.h"

int main(int argc, char ** argv) {
    if (argc < 2) return 2;
    bl::GgufModel g(argv[1]);
    std::map<uint32_t, int> seen;
    std::mt19937 rng(1);
    std::normal_distribution<float> nd;
    for (const auto & t : g.tensors()) {
        if (t.shape.size() != 2 || t.shape[0] % 256 || t.name.find("_exps") != std::string::npos || t.type_id == 30 || t.type_id == 0) continue;
        if (seen[t.type_id]++ >= 2 || !t.data) continue;
        const int K = static_cast<int>(t.shape[0]), M = std::min<int>(static_cast<int>(t.shape[1]), 2048), T = 4;
        const size_t rb = t.nbytes / t.shape[1];
        void * W; float *x, *y0, *y1;
        cudaMalloc(&W, rb * M); cudaMalloc(&x, 4ull * T * K); cudaMalloc(&y0, 4ull * T * M); cudaMalloc(&y1, 4ull * T * M);
        cudaMemcpy(W, t.data, rb * M, cudaMemcpyHostToDevice);
        std::vector<float> hx(static_cast<size_t>(T) * K);
        for (float & v : hx) v = nd(rng);
        cudaMemcpy(x, hx.data(), 4 * hx.size(), cudaMemcpyHostToDevice);
        bl::cuda::MvGroup gr;
        gr.K = K;
        gr.add(t.type_id, W, rb, M, y0);
        bl::cuda::set_matmul_q8(false);
        bl::cuda::matmul_group(gr, x, T, nullptr);
        gr.seg[0].y = y1;
        bl::cuda::set_matmul_q8(true);
        bl::cuda::matmul_group(gr, x, T, nullptr);
        cudaDeviceSynchronize();
        std::vector<float> a(static_cast<size_t>(T) * M), b(a.size());
        cudaMemcpy(a.data(), y0, 4 * a.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(b.data(), y1, 4 * b.size(), cudaMemcpyDeviceToHost);
        double num = 0, den = 0;
        for (size_t i = 0; i < a.size(); ++i) { num += (a[i] - b[i]) * (a[i] - b[i]); den += a[i] * a[i]; }
        std::printf("%-36s type %2u K %5d: rel err %.2e  %s\n", t.name.c_str(), t.type_id, K, std::sqrt(num / den),
                    std::sqrt(num / den) < 2e-2 ? "ok" : "BAD");
        cudaFree(W); cudaFree(x); cudaFree(y0); cudaFree(y1);
    }
    return 0;
}
