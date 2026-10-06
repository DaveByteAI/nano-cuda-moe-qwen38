// Qwen3.8-Flash-Next (`qwen4exp`) hyperparameters, read from GGUF metadata and checked against the shapes this
// engine is written for. See docs/MODEL.md.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "bl/gguf.h"

namespace bl {

struct ModelConfig {
    int n_layer = 0, n_embd = 0, n_vocab = 0;
    int n_head = 0, n_head_kv = 0, head_dim = 0;       // QSA
    int rope_dims = 0;
    std::array<int, 4> rope_sections{};
    double rope_freq_base = 0, rms_eps = 0;

    int n_expert = 0, n_expert_used = 0, n_ff_exp = 0, n_ff_shexp = 0;

    int ssm_conv = 0, ssm_state = 0, ssm_groups = 0, ssm_v_heads = 0, ssm_inner = 0;   // GDN
    int full_attn_interval = 0;

    int hc = 0, hc_low_rank = 0;                       // hyper-connections

    int idx_heads = 0, idx_dim = 0, idx_top_k = 0;     // QSA indexer
    std::vector<int> compress_ratio;                   // per layer, 0 on GDN layers

    int ple_layer = -1, ple_ngram = 0, ple_heads_per_ngram = 0, ple_conv = 0, ple_dim = 0;
    int ple_eos = 0, ple_image_token = 0;
    std::vector<uint64_t> ple_multipliers, ple_head_offsets, ple_head_vocab;

    int64_t context_length = 0;
    int bos = 0, eos = 0;

    bool is_qsa(int il) const { return (il + 1) % full_attn_interval == 0; }
    int  ple_heads() const { return (ple_ngram - 1) * ple_heads_per_ngram; }
    int  hc_dim() const { return hc * n_embd; }

    static ModelConfig from_gguf(const GgufModel & g);   // throws on anything this engine does not implement
};

}  // namespace bl
