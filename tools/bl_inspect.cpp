// bl-inspect: print the model configuration and the byte budget of a GGUF model (header only).
//   bl-inspect models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
#include <cstdio>
#include <map>
#include <string>

#include "bl/config.h"
#include "bl/gguf.h"

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s SHARD1.gguf\n", argv[0]);
        return 2;
    }
    try {
        bl::GgufModel g(argv[1]);
        const bl::ModelConfig c = bl::ModelConfig::from_gguf(g);
        std::printf("%s: %zu shard(s), %zu tensors, data %s\n", g.get_str("general.name").c_str(), g.n_shards(),
                    g.tensors().size(), g.complete() ? "complete" : "INCOMPLETE (partial download)");
        std::printf("layers %d (QSA every %d), embd %d, vocab %d, hc %dx (rank %d)\n", c.n_layer, c.full_attn_interval,
                    c.n_embd, c.n_vocab, c.hc, c.hc_low_rank);
        std::printf("experts %d top-%d ff %d, shared ff %d; GDN %d k-heads %d v-heads x %d; QSA %d/%d heads x %d\n",
                    c.n_expert, c.n_expert_used, c.n_ff_exp, c.n_ff_shexp, c.ssm_groups, c.ssm_v_heads, c.ssm_state,
                    c.n_head, c.n_head_kv, c.head_dim);
        std::printf("indexer %d x %d top-%d; PLE layer %d, %d heads x %d\n", c.idx_heads, c.idx_dim, c.idx_top_k,
                    c.ple_layer, c.ple_heads(), c.ple_dim);

        std::map<std::string, double> by;
        double experts = 0, dense = 0;
        size_t missing = 0;
        for (const auto & t : g.tensors()) {
            if (!t.data) ++missing;
            const bool exp = t.name.find("_exps.") != std::string::npos;
            if (exp) experts += t.nbytes;
            else if (t.name != "token_embd.weight" && t.name != "per_layer_token_embd.weight") dense += t.nbytes;
            by[bl::type_geometry(t.type_id).name] += t.nbytes;
        }
        std::printf("routed experts %.2f GB (%.3f GB per token); dense read per token %.3f GB\n", experts / 1e9,
                    experts / c.n_layer / c.n_expert * c.n_expert_used * c.n_layer / 1e9, dense / 1e9);
        std::printf("bytes by type:");
        for (const auto & [k, v] : by) std::printf(" %s=%.2fGB", k.c_str(), v / 1e9);
        std::printf("\ntensors whose bytes are not on disk yet: %zu\n", missing);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
