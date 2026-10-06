// bl-ref-dump: run llama.cpp's qwen4exp forward on a token list and save every named intermediate tensor.
// This is boundless's numerical reference: each node keeps the name qwen4exp.cpp gives it ("hc_mixed-12",
// "linear_attn_out-12", "ffn_moe_out-12", "l_last-12", "result_output", ...).
//
//   bl-ref-dump --model SHARD1.gguf --tokens-file prompt.tokens --out DIR [--filter REGEX] [--gpu-layers N]
//
// Output: DIR/index.tsv (name, type, ne0..ne3, byte offset, byte count) and DIR/data.bin (row-major, ne0 fastest,
// f32 for float tensors, i32 for index tensors). The default is the CPU backend: the most exact reference.
#include <cstdio>
#include <cstring>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

namespace {

struct Dump {
    std::regex          filter;
    std::ofstream       data, index;
    uint64_t            offset = 0;
    int                 graph  = 0;   // a prompt split into several ubatches evaluates several graphs
    std::vector<uint8_t> raw;
};

float fp16_to_f32(uint16_t h) { return ggml_fp16_to_fp32(h); }

bool on_tensor(ggml_tensor * t, bool ask, void * ud) {
    auto * d = static_cast<Dump *>(ud);
    const bool wanted = (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_I32) &&
                        t->name[0] != '\0' && std::regex_search(t->name, d->filter);
    if (ask) return wanted;
    if (!wanted) return true;

    const size_t n = ggml_nbytes(t);
    d->raw.resize(n);
    ggml_backend_tensor_get(t, d->raw.data(), 0, n);

    // gather with the strides: views ("Qcur_reshaped", stream slices) are not contiguous
    const int64_t ne0 = t->ne[0], ne1 = t->ne[1], ne2 = t->ne[2], ne3 = t->ne[3];
    std::vector<uint32_t> out(static_cast<size_t>(ne0 * ne1 * ne2 * ne3));
    size_t k = 0;
    for (int64_t i3 = 0; i3 < ne3; ++i3)
        for (int64_t i2 = 0; i2 < ne2; ++i2)
            for (int64_t i1 = 0; i1 < ne1; ++i1)
                for (int64_t i0 = 0; i0 < ne0; ++i0) {
                    const uint8_t * p = d->raw.data() + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
                    uint32_t v;
                    if (t->type == GGML_TYPE_F16) {
                        uint16_t h;
                        std::memcpy(&h, p, 2);
                        const float f = fp16_to_f32(h);
                        std::memcpy(&v, &f, 4);
                    } else {
                        std::memcpy(&v, p, 4);
                    }
                    out[k++] = v;
                }
    const size_t bytes = out.size() * 4;
    d->data.write(reinterpret_cast<const char *>(out.data()), static_cast<std::streamsize>(bytes));
    d->index << t->name << '\t' << (t->type == GGML_TYPE_I32 ? "i32" : "f32") << '\t' << ne0 << '\t' << ne1 << '\t'
             << ne2 << '\t' << ne3 << '\t' << d->offset << '\t' << bytes << '\t' << d->graph << '\n';
    d->offset += bytes;
    return true;
}

std::vector<llama_token> read_tokens(const std::string & path) {
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    std::string s = ss.str();
    for (char & c : s)
        if (c == ',') c = ' ';
    std::istringstream is(s);
    std::vector<llama_token> v;
    for (long x; is >> x;) v.push_back(static_cast<llama_token>(x));
    return v;
}

}  // namespace

int main(int argc, char ** argv) {
    std::string model, tokens_file, out_dir, filter = ".";
    int gpu_layers = 0, threads = 8, last = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--model") model = next();
        else if (a == "--tokens-file") tokens_file = next();
        else if (a == "--out") out_dir = next();
        else if (a == "--filter") filter = next();
        else if (a == "--gpu-layers") gpu_layers = std::stoi(next());
        else if (a == "--threads") threads = std::stoi(next());
        else if (a == "--last") last = std::stoi(next());   // logits only for the last N positions
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (model.empty() || tokens_file.empty() || out_dir.empty()) {
        std::fprintf(stderr, "usage: %s --model SHARD1.gguf --tokens-file F --out DIR [--filter RE] [--gpu-layers N]\n", argv[0]);
        return 2;
    }
    const std::vector<llama_token> toks = read_tokens(tokens_file);
    if (toks.empty()) { std::fprintf(stderr, "no tokens in %s\n", tokens_file.c_str()); return 1; }

    Dump d;
    d.filter = std::regex(filter);
    d.data.open(out_dir + "/data.bin", std::ios::binary);
    d.index.open(out_dir + "/index.tsv");
    if (!d.data || !d.index) { std::fprintf(stderr, "cannot write into %s\n", out_dir.c_str()); return 1; }
    d.index << "name\ttype\tne0\tne1\tne2\tne3\toffset\tbytes\tgraph\n";

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = gpu_layers;
    llama_model * m = llama_model_load_from_file(model.c_str(), mp);
    if (!m) { std::fprintf(stderr, "cannot load %s\n", model.c_str()); return 1; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = cp.n_batch = cp.n_ubatch = static_cast<uint32_t>(std::max<size_t>(toks.size(), 256));
    cp.n_threads = cp.n_threads_batch = threads;
    cp.cb_eval = on_tensor;
    cp.cb_eval_user_data = &d;
    llama_context * ctx = llama_init_from_model(m, cp);
    if (!ctx) { std::fprintf(stderr, "cannot create a context\n"); return 1; }

    // logits at every position, so a run can be compared teacher-forced (KL, top-1) and not only at the last token
    llama_batch b = llama_batch_init(static_cast<int32_t>(toks.size()), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        b.token[i]     = toks[i];
        b.pos[i]       = static_cast<llama_pos>(i);
        b.n_seq_id[i]  = 1;
        b.seq_id[i][0] = 0;
        b.logits[i]    = last <= 0 || i + static_cast<size_t>(last) >= toks.size();
    }
    b.n_tokens = static_cast<int32_t>(toks.size());
    if (llama_decode(ctx, b) != 0) { std::fprintf(stderr, "llama_decode failed\n"); return 1; }
    llama_batch_free(b);
    std::fprintf(stderr, "dumped %.1f MB for %zu tokens into %s\n", d.offset / 1e6, toks.size(), out_dir.c_str());

    llama_free(ctx);
    llama_model_free(m);
    llama_backend_free();
    return 0;
}
