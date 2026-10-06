// Time the matvec of every dense matrix of one GDN and one QSA layer (and the head), as the engine runs them.
//   bench_dense SHARD1.gguf [T]      (T > 1: the multi-token matmul of a verify window)
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "bl/cuda_fused.h"
#include "bl/cuda_quant.h"
#include "bl/gguf.h"

int main(int argc, char ** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s SHARD1.gguf\n", argv[0]); return 2; }
    bl::GgufModel g(argv[1]);
    const int T = argc > 2 ? std::atoi(argv[2]) : 1;
    std::vector<std::string> names;
    for (int il : {0, 3})
        for (const auto & t : g.tensors())
            if (t.name.rfind("blk." + std::to_string(il) + ".", 0) == 0 && t.shape.size() == 2 &&
                t.name.find("_exps") == std::string::npos && t.shape[0] % 8 == 0 && t.shape[0] >= 64)
                names.push_back(t.name);
    names.push_back("output.weight");

    float *x, *y;
    cudaMalloc(&x, sizeof(float) * 16384 * 4);
    cudaMalloc(&y, sizeof(float) * 300000 * 4);
    {   // non-zero activations (the q8 path's scales)
        std::vector<float> h(16384 * 4);
        for (size_t i = 0; i < h.size(); ++i) h[i] = 0.01f * static_cast<float>((i * 7919) % 201) - 1.f;
        cudaMemcpy(x, h.data(), sizeof(float) * h.size(), cudaMemcpyHostToDevice);
    }
    auto run = [&](const bl::cuda::MvGroup & gr) {
        if (T > 1) bl::cuda::matmul_group(gr, x, T, nullptr);
        else bl::cuda::matvec_group(gr, x, nullptr);
    };
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    double total_ms = 0, total_mb = 0;
    std::printf("%-36s %-8s %6s %7s %9s %9s %8s\n", "tensor", "type", "K", "M", "MB", "us", "GB/s");
    for (const auto & n : names) {
        const bl::Tensor & t = g.at(n);
        const int K = static_cast<int>(t.shape[0]), M = static_cast<int>(t.elements() / K);
        const size_t rb = bl::row_bytes(t.type_id, K);
        void * W;
        cudaMalloc(&W, t.nbytes);
        cudaMemcpy(W, t.data, t.nbytes, cudaMemcpyHostToDevice);
        bl::cuda::MvGroup g1;   // the engine's path: one segment of the grouped kernel
        g1.K = K;
        g1.add(t.type_id, W, rb, M, y);
        for (int i = 0; i < 5; ++i) run(g1);
        cudaEventRecord(e0);
        const int it = 50;
        for (int i = 0; i < it; ++i) run(g1);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        const double us = ms * 1e3 / it;
        std::printf("%-36s %-8s %6d %7d %9.2f %9.1f %8.0f\n", n.c_str(), bl::type_geometry(t.type_id).name, K, M,
                    t.nbytes / 1e6, us, t.nbytes / us / 1e3);
        total_ms += us / 1e3;
        total_mb += t.nbytes / 1e6;
        cudaFree(W);
    }
    {   // the hyper-connection group as the engine runs it: down (320 rows) + inject (4 rows), K = 10240
        const bl::Tensor & td = g.at("blk.3.hc_attn_down.weight"), & ti = g.at("blk.3.hc_attn_inject.weight");
        constexpr int kCopies = 12;   // rotated: the L2 (6 MB) cannot keep a copy warm, as in the engine (a layer each)
        void *Wd[kCopies], *Wi[kCopies];
        bl::cuda::MvGroup gr[kCopies];
        for (int c = 0; c < kCopies; ++c) {
            cudaMalloc(&Wd[c], td.nbytes); cudaMalloc(&Wi[c], ti.nbytes);
            cudaMemcpy(Wd[c], td.data, td.nbytes, cudaMemcpyHostToDevice); cudaMemcpy(Wi[c], ti.data, ti.nbytes, cudaMemcpyHostToDevice);
            gr[c].K = 10240;
            gr[c].add(td.type_id, Wd[c], 10240 * 2, 320, y);
            gr[c].add(ti.type_id, Wi[c], 10240 * 2, 4, y + 320 * 4);
        }
        auto run_mm = [&](const bl::cuda::MvGroup & q) { bl::cuda::matmul_group(q, x, T, nullptr); };   // the engine's call
        for (int i = 0; i < 5; ++i) run_mm(gr[i % kCopies]);
        cudaEventRecord(e0);
        for (int i = 0; i < 200; ++i) run_mm(gr[i % kCopies]);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        const double us = ms * 1e3 / 200;
        std::printf("%-36s %-8s %6d %7d %9.2f %9.1f %8.0f   (cold L2)\n", "hc_attn down+inject (one group)", "BF16", 10240, 324,
                    (td.nbytes + ti.nbytes) / 1e6, us, (td.nbytes + ti.nbytes) / us / 1e3);
        for (int c = 0; c < kCopies; ++c) { cudaFree(Wd[c]); cudaFree(Wi[c]); }
    }
    {   // hc_up_mix (W_up [hc*D][R] bf16, R = 320), rotated copies (cold L2)
        const bl::Tensor & tu = g.at("blk.3.hc_attn_up.weight");
        constexpr int kCopies = 12;
        void * Wu[kCopies];
        for (int c = 0; c < kCopies; ++c) { cudaMalloc(&Wu[c], tu.nbytes); cudaMemcpy(Wu[c], tu.data, tu.nbytes, cudaMemcpyHostToDevice); }
        float *lo, *xn, *mixed;
        cudaMalloc(&lo, sizeof(float) * 4 * 320); cudaMalloc(&xn, sizeof(float) * 4 * 10240); cudaMalloc(&mixed, sizeof(float) * 4 * 2560);
        cudaMemset(lo, 0, sizeof(float) * 4 * 320); cudaMemset(xn, 0, sizeof(float) * 4 * 10240);
        for (int i = 0; i < 5; ++i) bl::cuda::hc_up_mix_t(Wu[i % kCopies], lo, 320, xn, mixed, 2560, 4, T, nullptr);
        cudaEventRecord(e0);
        for (int i = 0; i < 200; ++i) bl::cuda::hc_up_mix_t(Wu[i % kCopies], lo, 320, xn, mixed, 2560, 4, T, nullptr);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        const double us = ms * 1e3 / 200;
        std::printf("%-36s %-8s %6d %7d %9.2f %9.1f %8.0f   (cold L2)\n", "hc_up_mix (fused)", "BF16", 320, 10240, tu.nbytes / 1e6, us,
                    tu.nbytes / us / 1e3);
        for (int c = 0; c < kCopies; ++c) cudaFree(Wu[c]);
        cudaFree(lo); cudaFree(xn); cudaFree(mixed);
    }
    std::printf("total %.2f ms for %.1f MB (%.0f GB/s); the L2 (6 MB) holds the small ones between repeats\n", total_ms,
                total_mb, total_mb / total_ms);
    return 0;
}
