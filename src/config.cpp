#include "bl/config.h"

#include <stdexcept>
#include <string>

namespace bl {

namespace {

void require(bool ok, const std::string & what) {
    if (!ok) throw std::runtime_error("unsupported model: " + what);
}

}  // namespace

ModelConfig ModelConfig::from_gguf(const GgufModel & g) {
    require(g.get_str("general.architecture") == "qwen4exp", "architecture is not qwen4exp");
    const std::string a = "qwen4exp.";
    auto I = [&](const std::string & k) { return static_cast<int>(g.get_int(a + k)); };

    ModelConfig c;
    c.n_layer   = I("block_count");
    c.n_embd    = I("embedding_length");
    c.n_head    = I("attention.head_count");
    c.n_head_kv = I("attention.head_count_kv");
    c.head_dim  = I("attention.key_length");
    require(I("attention.value_length") == c.head_dim, "key and value head sizes differ");
    c.rope_dims = I("rope.dimension_count");
    const auto sec = g.get_int_array(a + "rope.dimension_sections");
    require(sec.size() == 4, "rope.dimension_sections needs 4 entries");
    for (int i = 0; i < 4; ++i) c.rope_sections[i] = static_cast<int>(sec[i]);
    c.rope_freq_base = g.get_float(a + "rope.freq_base");
    c.rms_eps        = g.get_float(a + "attention.layer_norm_rms_epsilon");

    c.n_expert      = I("expert_count");
    c.n_expert_used = I("expert_used_count");
    c.n_ff_exp      = I("expert_feed_forward_length");
    c.n_ff_shexp    = I("expert_shared_feed_forward_length");

    c.ssm_conv    = I("ssm.conv_kernel");
    c.ssm_state   = I("ssm.state_size");
    c.ssm_groups  = I("ssm.group_count");
    c.ssm_v_heads = I("ssm.time_step_rank");
    c.ssm_inner   = I("ssm.inner_size");
    require(c.ssm_v_heads * c.ssm_state == c.ssm_inner, "ssm inner size != v heads * state");
    require(c.ssm_v_heads % c.ssm_groups == 0, "ssm v heads not a multiple of k heads");
    c.full_attn_interval = g.has(a + "full_attention_interval") ? I("full_attention_interval") : 4;
    require(!g.has(a + "attention.recurrent_layers"), "explicit recurrent layer list");

    c.hc          = I("hyper_connection.count");
    c.hc_low_rank = I("hyper_connection.low_rank");
    require(c.hc > 1, "hyper_connection.count must be > 1");

    c.idx_heads = I("attention.indexer.head_count");
    c.idx_dim   = I("attention.indexer.key_length");
    c.idx_top_k = I("attention.indexer.top_k");
    for (int64_t r : g.get_int_array(a + "attention.compress_ratios")) c.compress_ratio.push_back(static_cast<int>(r));
    require(static_cast<int>(c.compress_ratio.size()) == c.n_layer, "compress_ratios length != block_count");
    for (int il = 0; il < c.n_layer; ++il)
        require((c.compress_ratio[il] > 0) == c.is_qsa(il), "compress ratio set on a GDN layer or missing on QSA");

    const auto ple_layers = g.get_int_array(a + "ple.layers");
    require(ple_layers.size() == 1, "exactly one PLE layer");
    c.ple_layer           = static_cast<int>(ple_layers[0]);
    require(!c.is_qsa(c.ple_layer), "PLE layer must be a linear-attention layer");
    c.ple_ngram           = I("ple.ngram_size");
    c.ple_heads_per_ngram = I("ple.heads_per_ngram");
    c.ple_conv            = I("ple.conv_kernel");
    c.ple_dim             = I("embedding_length_per_layer_input");
    c.ple_eos             = I("ple.eos_token_id");
    c.ple_image_token     = g.has(a + "ple.image_token_id") ? I("ple.image_token_id") : c.ple_eos;
    auto u64 = [&](const std::string & k, size_t n) {
        std::vector<uint64_t> v;
        for (int64_t x : g.get_int_array(a + k)) v.push_back(static_cast<uint64_t>(x));
        require(v.size() >= n, k + " too short");
        v.resize(n);
        return v;
    };
    c.ple_multipliers  = u64("ple.layer_multipliers", c.ple_ngram);
    c.ple_head_offsets = u64("ple.head_offsets", c.ple_heads());
    c.ple_head_vocab   = u64("ple.head_vocab_sizes", c.ple_heads());

    c.context_length = g.get_int(a + "context_length");
    c.bos            = static_cast<int>(g.get_int("tokenizer.ggml.bos_token_id"));
    c.eos            = static_cast<int>(g.get_int("tokenizer.ggml.eos_token_id"));

    const Tensor & emb = g.at("token_embd.weight");
    require(emb.shape.size() == 2 && emb.shape[0] == c.n_embd, "token_embd shape");
    c.n_vocab = static_cast<int>(emb.shape[1]);

    // the shapes the kernels assume (docs/MODEL.md); a different checkpoint fails here, not mid-forward
    auto shape = [&](const std::string & n, std::vector<int64_t> s) {
        const Tensor & t = g.at(n);
        require(t.shape == s, n + " has an unexpected shape");
    };
    const int64_t E = c.n_embd, HD = c.hc_dim();
    for (int il = 0; il < c.n_layer; ++il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        shape(p + "hc_attn_down.weight", {HD, c.hc_low_rank});
        shape(p + "hc_attn_up.weight", {c.hc_low_rank, HD});
        shape(p + "hc_attn_inject.weight", {HD, c.hc});
        shape(p + "ffn_gate_inp.weight", {E, c.n_expert});
        shape(p + "ffn_gate_exps.weight", {E, c.n_ff_exp, c.n_expert});
        shape(p + "ffn_up_exps.weight", {E, c.n_ff_exp, c.n_expert});
        shape(p + "ffn_down_exps.weight", {c.n_ff_exp, E, c.n_expert});
        if (c.is_qsa(il)) {
            shape(p + "attn_q.weight", {E, 2LL * c.n_head * c.head_dim});
            shape(p + "attn_k.weight", {E, 1LL * c.n_head_kv * c.head_dim});
            shape(p + "attn_v.weight", {E, 1LL * c.n_head_kv * c.head_dim});
            shape(p + "indexer.q_proj.weight", {E, 1LL * c.idx_heads * c.idx_dim});
            shape(p + "indexer.k_proj.weight", {E, c.idx_dim});
        } else {
            const int64_t conv_dim = 2LL * c.ssm_groups * c.ssm_state + c.ssm_inner;
            shape(p + "attn_qkv.weight", {E, conv_dim});
            shape(p + "attn_gate.weight", {E, c.ssm_inner});
            shape(p + "ssm_conv1d.weight", {c.ssm_conv, conv_dim});
            shape(p + "ssm_out.weight", {c.ssm_inner, E});
        }
    }
    return c;
}

}  // namespace bl
