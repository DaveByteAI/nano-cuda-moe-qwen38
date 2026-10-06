// bl-mtp-pack: the MTP layer's tensors (downloaded by tools/fetch_mtp.py) -> the GGUF the engine loads.
//   bl-mtp-pack models/mtp-src models/mtp-q2_0.gguf
// The 512 routed experts become Q2_0 (blocks of 64 values: an fp16 scale d and 2-bit codes q in {-1, 0, 1, 2}, value
// q * d), with d chosen per block for the least squared error; the other matrices stay BF16 as in the checkpoint
// (dimensions reversed to ggml's order), the 1-D norm weights become F32 (the engine adds the 1 of the (1 + w) norm).
// Experts are stored as rows of their input dimension, gate_up gate-first, as the engine expects.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "ggml.h"
#include "gguf.h"
#include "nlohmann/json.hpp"

namespace {

constexpr int kBlock = 64;   // QK2_0
struct BlockQ2 {
    uint16_t d;
    uint8_t  qs[kBlock / 4];
};
static_assert(sizeof(BlockQ2) == 18, "block_q2_0");

float bf16(uint16_t v) {
    const uint32_t u = static_cast<uint32_t>(v) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// the squared error of x quantized with scale d, and the codes (q + 1 in 0..3)
float q2_error(const float * x, float d, uint8_t * code) {
    const float id = d > 0.f ? 1.f / d : 0.f;
    float e = 0.f;
    for (int j = 0; j < kBlock; ++j) {
        const int q = std::clamp(static_cast<int>(std::lround(x[j] * id)), -1, 2);
        if (code) code[j] = static_cast<uint8_t>(q + 1);
        const float r = x[j] - q * d;
        e += r * r;
    }
    return e;
}

// one block: scale candidates amax * f (f = 0.30 .. 1.00), each refined once by least squares for its codes; the
// best d is rounded to fp16 and the codes are taken again with that d
BlockQ2 quantize_block(const float * x) {
    float amax = 0.f;
    for (int j = 0; j < kBlock; ++j) amax = std::max(amax, std::fabs(x[j]));
    BlockQ2 b{};
    if (amax == 0.f) {
        std::memset(b.qs, 0x55, sizeof b.qs);   // every code 1: value 0
        return b;
    }
    float best_d = amax, best_e = q2_error(x, amax, nullptr);
    uint8_t code[kBlock];
    for (int k = 0; k <= 35; ++k) {
        const float d0 = amax * (0.30f + 0.02f * k);
        float e = q2_error(x, d0, code);
        if (e < best_e) best_e = e, best_d = d0;
        float sxq = 0.f, sqq = 0.f;   // the least-squares scale for these codes
        for (int j = 0; j < kBlock; ++j) {
            const float q = static_cast<float>(code[j]) - 1.f;
            sxq += x[j] * q;
            sqq += q * q;
        }
        if (sqq > 0.f && sxq > 0.f) {
            const float d1 = sxq / sqq;
            e = q2_error(x, d1, nullptr);
            if (e < best_e) best_e = e, best_d = d1;
        }
    }
    b.d = ggml_fp32_to_fp16(best_d);
    q2_error(x, ggml_fp16_to_fp32(b.d), code);
    for (int j = 0; j < kBlock; ++j) b.qs[j / 4] |= static_cast<uint8_t>(code[j] << (2 * (j % 4)));
    return b;
}

std::vector<uint8_t> read_file(const std::string & path, size_t want) {
    std::ifstream f(path, std::ios::binary);
    std::vector<uint8_t> v(want);
    if (!f || !f.read(reinterpret_cast<char *>(v.data()), static_cast<std::streamsize>(want)) || f.peek() != EOF)
        throw std::runtime_error(path + ": not " + std::to_string(want) + " bytes");
    return v;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s SRC_DIR (tools/fetch_mtp.py's) OUT.gguf\n", argv[0]);
        return 2;
    }
    try {
        const std::string src = argv[1];
        std::ifstream mf(src + "/manifest.json");
        if (!mf) throw std::runtime_error(src + "/manifest.json: not found (run tools/fetch_mtp.py first)");
        const nlohmann::json man = nlohmann::json::parse(mf);

        ggml_init_params ip{static_cast<size_t>(64) << 20, nullptr, true};
        ggml_context * ctx = ggml_init(ip);
        gguf_context * gg = gguf_init_empty();
        gguf_set_val_str(gg, "general.architecture", "qwen4exp-mtp");
        const std::string source = man["repo"].get<std::string>() + "@" + man["revision"].get<std::string>() + ", mtp.* tensors";
        gguf_set_val_str(gg, "mtp.source", source.c_str());
        gguf_set_val_str(gg, "mtp.expert_format", "q2_0");
        gguf_set_val_str(gg, "mtp.expert_quantizer", "per-block least-squares scale search (bl-mtp-pack)");

        std::vector<std::vector<uint8_t>> keep;   // the tensors' bytes, alive until the file is written
        double sq_err = 0, sq_val = 0;
        for (const auto & t : man["tensors"]) {
            const std::string name = t["name"], dtype = t["dtype"];
            std::vector<int64_t> shape = t["shape"];
            if (dtype != "BF16") throw std::runtime_error(name + ": " + dtype + ", expected BF16");
            size_t n = 1;
            for (int64_t s : shape) n *= static_cast<size_t>(s);
            std::vector<uint8_t> raw = read_file(src + "/" + t["file"].get<std::string>(), 2 * n);
            const uint16_t * w = reinterpret_cast<const uint16_t *>(raw.data());
            std::vector<int64_t> ne(shape.rbegin(), shape.rend());   // ggml order: innermost first
            ggml_tensor * gt = nullptr;
            const bool experts = name.find(".mlp.experts.") != std::string::npos;
            if (experts) {   // [E][out][in] rows of `in` values -> Q2_0; [E][in][out] is transposed first
                if (shape.size() != 3) throw std::runtime_error(name + ": not 3-D");
                const bool gate_up = name.find("gate_up") != std::string::npos;
                const int64_t E = shape[0];
                int64_t rows = shape[1], in = shape[2];
                bool transpose = false;
                if (gate_up ? (in == 2 * 640) : (in == 2560)) std::swap(rows, in), transpose = true;   // the checkpoint's order
                if (in % kBlock) throw std::runtime_error(name + ": rows not a multiple of 64");
                std::vector<uint8_t> out(static_cast<size_t>(E) * rows * (in / kBlock) * sizeof(BlockQ2));
                auto * ob = reinterpret_cast<BlockQ2 *>(out.data());
                std::atomic<int64_t> next{0};
                std::vector<double> err(std::thread::hardware_concurrency()), val(err.size());
                std::vector<std::thread> pool;
                for (size_t th = 0; th < err.size(); ++th)
                    pool.emplace_back([&, th] {
                        std::vector<float> row(static_cast<size_t>(in));
                        for (int64_t r; (r = next.fetch_add(1)) < E * rows;) {   // r: expert * rows + row
                            const int64_t e = r / rows, o = r % rows;
                            const uint16_t * we = w + static_cast<size_t>(e) * rows * in;
                            for (int64_t i = 0; i < in; ++i) row[i] = bf16(transpose ? we[i * rows + o] : we[o * in + i]);
                            for (int64_t blk = 0; blk < in / kBlock; ++blk) {
                                const BlockQ2 b = quantize_block(row.data() + blk * kBlock);
                                ob[static_cast<size_t>(r) * (in / kBlock) + blk] = b;
                                const float d = ggml_fp16_to_fp32(b.d);
                                for (int j = 0; j < kBlock; ++j) {
                                    const float x = row[blk * kBlock + j], q = ((b.qs[j / 4] >> (2 * (j % 4))) & 3) - 1.f;
                                    err[th] += (x - q * d) * (x - q * d);
                                    val[th] += x * x;
                                }
                            }
                        }
                    });
                for (auto & p : pool) p.join();
                for (size_t th = 0; th < err.size(); ++th) sq_err += err[th], sq_val += val[th];
                const int64_t qne[3] = {in, rows, E};
                gt = ggml_new_tensor(ctx, GGML_TYPE_Q2_0, 3, qne);
                keep.push_back(std::move(out));
                std::printf("%-58s Q2_0 [%lld, %lld, %lld]%s\n", name.c_str(), static_cast<long long>(in),
                            static_cast<long long>(rows), static_cast<long long>(E), transpose ? " (transposed)" : "");
            } else if (shape.size() == 1) {   // the norm weights: F32
                std::vector<uint8_t> out(4 * n);
                float * f = reinterpret_cast<float *>(out.data());
                for (size_t i = 0; i < n; ++i) f[i] = bf16(w[i]);
                gt = ggml_new_tensor(ctx, GGML_TYPE_F32, 1, ne.data());
                keep.push_back(std::move(out));
            } else {
                gt = ggml_new_tensor(ctx, GGML_TYPE_BF16, static_cast<int>(ne.size()), ne.data());
                keep.push_back(std::move(raw));
            }
            ggml_set_name(gt, name.c_str());
            gt->data = keep.back().data();
            gguf_add_tensor(gg, gt);
        }
        std::printf("experts: relative RMS error of Q2_0 %.4f\n", std::sqrt(sq_err / sq_val));
        if (!gguf_write_to_file(gg, argv[2], false)) throw std::runtime_error(std::string("cannot write ") + argv[2]);
        std::printf("wrote %s (%zu tensors)\n", argv[2], keep.size());
        gguf_free(gg);
        ggml_free(ctx);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "bl-mtp-pack: %s\n", e.what());
        return 1;
    }
    return 0;
}
