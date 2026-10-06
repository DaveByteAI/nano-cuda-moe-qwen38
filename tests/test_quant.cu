// Parity of bl::cuda's dequantization and matvec against ggml, on real rows of the model file.
//   test_quant SHARD1.gguf [rows]
// For every weight format in the file it takes the first tensor whose bytes are on disk and checks:
//   dequant: our GPU values vs ggml's to_float (CPU) - bit-identical expected, max ulp reported
//   matvec:  our GPU dot products vs a double-precision dot of ggml's values, relative to sum |w*x|
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "bl/cuda_quant.h"
#include "bl/gguf.h"
#include "ggml.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); return 1; } } while (0)

static int64_t ulp_diff(float a, float b) {
    int32_t ia, ib;
    std::memcpy(&ia, &a, 4);
    std::memcpy(&ib, &b, 4);
    if (ia < 0) ia = INT32_MIN - ia;
    if (ib < 0) ib = INT32_MIN - ib;
    return std::llabs(static_cast<int64_t>(ia) - ib);
}

int main(int argc, char ** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s SHARD1.gguf [rows]\n", argv[0]); return 2; }
    const bool bench = argc > 2 && std::strcmp(argv[2], "--bench") == 0;
    const int want_rows = bench ? 1 << 30 : (argc > 2 ? std::atoi(argv[2]) : 256);
    bl::GgufModel g(argv[1]);

    std::map<uint32_t, const bl::Tensor *> pick;
    for (const auto & t : g.tensors())
        if (t.data && t.shape.size() >= 2 && t.shape[0] % 64 == 0 && bl::cuda::supported(t.type_id) && !pick.count(t.type_id))
            pick[t.type_id] = &t;

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> U(-1.f, 1.f);
    int failures = 0;
    std::printf("%-8s %-40s %6s %6s  %-26s %s\n", "type", "tensor", "K", "rows", "dequant (exact / max ulp)", "matvec max rel err");
    for (const auto & [type, t] : pick) {
        const int K = static_cast<int>(t->shape[0]);
        const int M = static_cast<int>(std::min<int64_t>(want_rows, t->elements() / K));
        const size_t rb = bl::row_bytes(type, K);
        const ggml_type_traits * tr = ggml_get_type_traits(static_cast<ggml_type>(type));

        std::vector<float> ref(static_cast<size_t>(M) * K);
        for (int r = 0; r < M; ++r) {
            float * dst = ref.data() + static_cast<size_t>(r) * K;
            if (type == GGML_TYPE_F32) std::memcpy(dst, t->data + r * rb, rb);   // ggml has no to_float for F32
            else tr->to_float(t->data + r * rb, dst, K);
        }

        std::vector<float> x(K);
        for (auto & v : x) v = U(rng);

        void * dW; float *dOut, *dx, *dy;
        CK(cudaMalloc(&dW, rb * M));
        CK(cudaMalloc(&dOut, sizeof(float) * M * K));
        CK(cudaMalloc(&dx, sizeof(float) * K));
        CK(cudaMalloc(&dy, sizeof(float) * M));
        CK(cudaMemcpy(dW, t->data, rb * M, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dx, x.data(), sizeof(float) * K, cudaMemcpyHostToDevice));
        bl::cuda::dequant_rows(type, dW, rb, M, K, dOut, nullptr);
        bl::cuda::matvec(type, dW, rb, M, K, dx, dy, nullptr);
        CK(cudaDeviceSynchronize());
        std::vector<float> got(static_cast<size_t>(M) * K), y(M);
        CK(cudaMemcpy(got.data(), dOut, sizeof(float) * M * K, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(y.data(), dy, sizeof(float) * M, cudaMemcpyDeviceToHost));

        size_t exact = 0;
        int64_t max_ulp = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            if (std::memcmp(&got[i], &ref[i], 4) == 0) ++exact;
            else max_ulp = std::max(max_ulp, ulp_diff(got[i], ref[i]));
        }
        double max_rel = 0;
        for (int r = 0; r < M; ++r) {
            double s = 0, sa = 0;
            for (int k = 0; k < K; ++k) {
                const double p = static_cast<double>(ref[static_cast<size_t>(r) * K + k]) * x[k];
                s += p;
                sa += std::fabs(p);
            }
            if (sa > 0) max_rel = std::max(max_rel, std::fabs(y[r] - s) / sa);
        }
        float gbps = 0;
        if (bench) {   // whole tensor, weights cold in L2 (6 MB) for anything but the smallest tensors
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0); cudaEventCreate(&e1);
            for (int i = 0; i < 3; ++i) bl::cuda::matvec(type, dW, rb, M, K, dx, dy, nullptr);
            cudaEventRecord(e0);
            for (int i = 0; i < 20; ++i) bl::cuda::matvec(type, dW, rb, M, K, dx, dy, nullptr);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms = 0;
            cudaEventElapsedTime(&ms, e0, e1);
            gbps = static_cast<float>(rb * M * 20 / (ms * 1e6));
        }
        const bool ok = exact == got.size() && max_rel < 1e-5;
        failures += !ok;
        char dq[64];
        std::snprintf(dq, sizeof dq, "%zu/%zu / %lld", exact, got.size(), static_cast<long long>(max_ulp));
        std::printf("%-8s %-40s %6d %6d  %-26s %.2e %s", ggml_type_name(static_cast<ggml_type>(type)), t->name.c_str(),
                    K, M, dq, max_rel, ok ? "ok" : "FAIL");
        if (bench) std::printf("  matvec %.0f GB/s (%.1f MB)", gbps, rb * M / 1e6);
        std::printf("\n");
        cudaFree(dW); cudaFree(dOut); cudaFree(dx); cudaFree(dy);
    }
    std::printf("%s (%zu formats)\n", failures ? "FAILED" : "all formats match ggml", pick.size());
    return failures ? 1 : 0;
}
