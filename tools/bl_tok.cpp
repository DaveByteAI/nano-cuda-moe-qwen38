// The tokenizer from the command line (llama.cpp's, vocab only: the GGUF's weights are not read).
//   bl-tok SHARD1.gguf "text"                 check: chat template -> ids -> text, end-of-generation token
//   bl-tok SHARD1.gguf --encode FILE [--special]   FILE's text as ids, comma-separated (--special: <|im_start|> etc.
//                                                  are tokens, not text)
//   bl-tok SHARD1.gguf --decode FILE          comma-separated ids in FILE -> text
//   bl-tok SHARD1.gguf --chat FILE            FILE's text as one user message, rendered for the model (no thinking)
// (bench/*.py build the benchmark inputs with these.)
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

#include "bl/text.h"

namespace {

std::string slurp(const char * path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot read %s\n", path); std::exit(1); }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s SHARD1.gguf TEXT | --encode FILE [--special] | --decode FILE | --chat FILE\n", argv[0]);
        return 2;
    }
    const std::string mode = argv[2];
    if (mode == "--encode" || mode == "--decode" || mode == "--chat") {
        if (argc < 4) { std::fprintf(stderr, "%s needs a FILE\n", mode.c_str()); return 2; }
        if (mode == "--chat") {   // the template needs no tokenizer
            const std::string s = bl::Text::render({{"user", slurp(argv[3]), ""}}, true, "");
            std::fwrite(s.data(), 1, s.size(), stdout);
            return 0;
        }
        bl::Text tx(argv[1]);
        if (mode == "--encode") {
            const bool special = argc > 4 && !std::strcmp(argv[4], "--special");
            const auto ids = tx.tokenize(slurp(argv[3]), special);
            for (size_t i = 0; i < ids.size(); ++i) std::printf(i ? ",%d" : "%d", ids[i]);
            std::printf("\n");
        } else {
            std::vector<int> ids;
            std::stringstream ss(slurp(argv[3]));
            for (std::string v; std::getline(ss, v, ',');)
                if (v.find_first_not_of(" \n\r\t") != std::string::npos) ids.push_back(std::stoi(v));
            const std::string s = tx.detokenize(ids);
            std::fwrite(s.data(), 1, s.size(), stdout);
        }
        return 0;
    }

    const auto t0 = std::chrono::steady_clock::now();
    bl::Text tx(argv[1]);
    std::printf("tokenizer loaded in %.2f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    const std::string chat = bl::Text::render({{"system", "You are helpful.", ""}, {"user", argv[2], ""}}, true, "");
    const auto ids = tx.tokenize(chat);
    std::printf("%zu tokens:", ids.size());
    for (int id : ids) std::printf(" %d", id);
    std::printf("\nround trip %s\n", tx.detokenize(ids) == chat ? "exact" : "DIFFERS");
    std::printf("eog: %d (<|im_end|> = %d)\n", tx.is_eog(ids[ids.size() - 9]), ids[ids.size() - 9]);
    return 0;
}
