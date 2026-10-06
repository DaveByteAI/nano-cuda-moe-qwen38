// GPU routed-expert kernels in isolation: 10 cached experts of one layer per format pair, all hits.
//   bench_experts SHARD1.gguf [T] [slots]   (a verify window: T tokens choosing K = 10 each among `slots` experts)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "bl/cuda_pipeline.h"
#include "bl/gguf.h"

int main(int argc, char ** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s SHARD1.gguf\n", argv[0]); return 2; }
    bl::GgufModel g(argv[1]);
    const int D = 2560, F = 640, NE = 512, K = 10;
    const int T = argc > 2 ? std::atoi(argv[2]) : 1, N = argc > 3 ? std::atoi(argv[3]) : K;
    std::map<std::pair<uint32_t, uint32_t>, int> pick;
    for (int il = 0; il < 48; ++il)
        pick.emplace(std::make_pair(g.at("blk." + std::to_string(il) + ".ffn_gate_exps.weight").type_id,
                                    g.at("blk." + std::to_string(il) + ".ffn_down_exps.weight").type_id), il);
    float *x, *act, *out;
    cudaMalloc(&x, bl::cuda::kMaxT * D * 4);
    cudaMalloc(&act, bl::cuda::kMaxSlots * bl::cuda::kMaxT * F * 4);
    cudaMalloc(&out, bl::cuda::kMaxSlots * bl::cuda::kMaxT * D * 4);
    {
        std::vector<float> h(bl::cuda::kMaxT * D);
        for (size_t i = 0; i < h.size(); ++i) h[i] = 0.01f * static_cast<float>((i * 7919) % 201) - 1.f;
        cudaMemcpy(x, h.data(), h.size() * 4, cudaMemcpyHostToDevice);
    }
    void * xq;
    cudaMalloc(&xq, bl::cuda::q8_bytes(D) * bl::cuda::kMaxT);
    bl::cuda::quantize_q8(x, D, xq, nullptr, T);
    bl::cuda::ExpertPlan * plan_d;
    cudaMalloc(&plan_d, sizeof(bl::cuda::ExpertPlan));
    cudaEvent_t e0, e1, e2;
    cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2);
    double tot_ms = 0, tot_mb = 0;
    for (auto [types, il] : pick) {
        const std::string p = "blk." + std::to_string(il) + ".";
        const auto & gt = g.at(p + "ffn_gate_exps.weight");
        const auto & ut = g.at(p + "ffn_up_exps.weight");
        const auto & dt = g.at(p + "ffn_down_exps.weight");
        const size_t gu = gt.nbytes / NE, db = dt.nbytes / NE, eb = 2 * gu + db;
        // 128 experts in VRAM so repeats do not run from L2; each repeat picks the next N
        const int pool = 128;
        uint8_t * arena;
        cudaMalloc(&arena, eb * pool);
        std::vector<uint8_t> buf(eb);
        for (int e = 0; e < pool; ++e) {
            std::memcpy(buf.data(), gt.data + e * gu, gu);
            std::memcpy(buf.data() + gu, ut.data + e * gu, gu);
            std::memcpy(buf.data() + 2 * gu, dt.data + e * db, db);
            cudaMemcpy(arena + e * eb, buf.data(), eb, cudaMemcpyHostToDevice);
        }
        const int reps = 60;
        std::vector<bl::cuda::ExpertPlan> plans(reps);
        for (int r = 0; r < reps; ++r) {
            bl::cuda::ExpertPlan & pl = plans[r];
            std::memset(&pl, 0, sizeof pl);
            pl.T = T; pl.K = K; pl.n_slots = N;
            for (int i = 0; i < N; ++i) {
                pl.ids[i] = i; pl.kind[i] = bl::cuda::kHit; pl.mask[i] = 0;
                pl.base[i] = arena + ((r * N + i) % pool) * eb;
            }
            for (int t = 0; t < T; ++t)   // token t takes K slots, spread so that the union is all N
                for (int k = 0; k < K; ++k) {
                    const int sl = (t * K + k) % N;
                    pl.sel[t][k] = sl; pl.w[t][k] = 0.1f; pl.mask[sl] |= 1 << t;
                }
        }
        bl::cuda::ExpertPlan * pd;
        cudaMalloc(&pd, sizeof(bl::cuda::ExpertPlan) * reps);
        cudaMemcpy(pd, plans.data(), sizeof(bl::cuda::ExpertPlan) * reps, cudaMemcpyHostToDevice);
        float gu_ms = 0, d_ms = 0;
        for (int pass = 0; pass < 2; ++pass) {
            cudaEventRecord(e0);
            for (int r = 0; r < reps; ++r)
                bl::cuda::plan_gate_up(gt.type_id, pd + r, 1, T, gu, F, D, xq, act, nullptr);
            cudaEventRecord(e1);
            for (int r = 0; r < reps; ++r)
                bl::cuda::plan_down(dt.type_id, pd + r, 1, T, gu, F, D, act, out, nullptr);
            cudaEventRecord(e2);
            cudaEventSynchronize(e2);
            cudaEventElapsedTime(&gu_ms, e0, e1);
            cudaEventElapsedTime(&d_ms, e1, e2);
        }
        const double gu_b = 2.0 * gu * N * reps, d_b = 1.0 * db * N * reps;
        std::printf("layer %2d  gate/up %-8s %5.0f GB/s (%5.1f us)   down %-7s %5.0f GB/s (%5.1f us)\n", il,
                    bl::type_geometry(gt.type_id).name, gu_b / gu_ms / 1e6, gu_ms * 1e3 / reps,
                    bl::type_geometry(dt.type_id).name, d_b / d_ms / 1e6, d_ms * 1e3 / reps);
        tot_ms += gu_ms + d_ms;
        tot_mb += (gu_b + d_b) / 1e6;
        cudaFree(arena);
        cudaFree(pd);
    }
    std::printf("all: %.0f GB/s\n", tot_mb / tot_ms);
    return 0;
}
