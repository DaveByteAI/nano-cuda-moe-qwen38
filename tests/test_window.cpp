// Verify windows against one-token steps, on real text: the same logits position by position.
//   test_window SHARD1.gguf TOKENS_FILE [n_tokens]
//   1. reference: step() token by token
//   2. windows of T = 2, 3, 4, all accepted (verify + commit(T))
//   3. rejection: windows [s_i, s_i+1, junk, junk] with commit(2) - the rolled-back state must match
// The q8_1 matmul path is off here (steps are fp32). A window's kernels sum in a different order, so the logits match closely, not bit for bit; the argmax must agree.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "bl/cuda_fused.h"
#include "bl/engine.h"

namespace {

std::vector<int> read_tokens(const std::string & path) {
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    std::string s = ss.str();
    for (char & ch : s)
        if (ch == ',') ch = ' ';
    std::istringstream is(s);
    std::vector<int> v;
    for (int x; is >> x;) v.push_back(x);
    return v;
}

struct Cmp {
    double max_abs = 0, max_rel = 0;
    int argmax_diff = 0, n = 0;
    void add(const float * a, const float * b, int V) {
        int ia = 0, ib = 0;
        double num = 0, den = 0;
        for (int i = 0; i < V; ++i) {
            max_abs = std::max(max_abs, static_cast<double>(std::fabs(a[i] - b[i])));
            num += static_cast<double>(a[i] - b[i]) * (a[i] - b[i]);
            den += static_cast<double>(b[i]) * b[i];
            if (a[i] > a[ia]) ia = i;
            if (b[i] > b[ib]) ib = i;
        }
        max_rel = std::max(max_rel, std::sqrt(num / den));
        argmax_diff += ia != ib;
        ++n;
    }
    bool ok() const { return argmax_diff == 0 && max_rel < 2e-3; }
};

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s SHARD1.gguf TOKENS [n]\n", argv[0]); return 2; }
    std::vector<int> toks = read_tokens(argv[2]);
    const int N = std::min<int>(argc > 3 ? std::atoi(argv[3]) : 64, static_cast<int>(toks.size()));
    toks.resize(N);
    bl::cuda::set_matmul_q8(false);   // steps use fp32 activations: compare windows on the same footing
    bl::Engine eng(argv[1], 512);
    const int V = eng.config().n_vocab;

    std::vector<std::vector<float>> ref(N);
    eng.reset();
    for (int i = 0; i < N; ++i) ref[i] = eng.step(toks[i], true);
    std::printf("reference: %d one-token steps\n", N);

    bool all_ok = true;
    for (int T = 1; T <= 4; ++T) {
        eng.reset();
        Cmp cmp, by_t[4];
        for (int i = 0; i + T <= N; i += T) {
            eng.verify(&toks[i], T, true);
            const std::vector<float> & L = eng.logits();
            for (int t = 0; t < T; ++t) {
                cmp.add(L.data() + static_cast<size_t>(t) * V, ref[i + t].data(), V);
                by_t[t].add(L.data() + static_cast<size_t>(t) * V, ref[i + t].data(), V);
            }
            eng.commit(T);
        }
        std::printf("  by position in the window:");
        for (int t = 0; t < T; ++t) std::printf(" t%d max rel %.2e;", t, by_t[t].max_rel);
        std::printf("\n");
        std::printf("windows of %d, all accepted: %d positions, argmax differs at %d, max rel %.2e, max abs %.3f  %s\n", T,
                    cmp.n, cmp.argmax_diff, cmp.max_rel, cmp.max_abs, cmp.ok() ? "ok" : "FAIL");
        all_ok &= cmp.ok();
    }

    {   // rejection: two real tokens then two wrong ones, keep two
        eng.reset();
        Cmp cmp;
        for (int i = 0; i + 2 <= N; i += 2) {
            const int w[4] = {toks[i], toks[i + 1], 12345, 777};
            eng.verify(w, 4, true);
            const std::vector<float> & L = eng.logits();
            for (int t = 0; t < 2; ++t) cmp.add(L.data() + static_cast<size_t>(t) * V, ref[i + t].data(), V);
            eng.commit(2);
        }
        std::printf("windows of 4 keeping 2 (rollback): %d positions, argmax differs at %d, max rel %.2e, max abs %.3f  %s\n",
                    cmp.n, cmp.argmax_diff, cmp.max_rel, cmp.max_abs, cmp.ok() ? "ok" : "FAIL");
        all_ok &= cmp.ok();
    }
    std::printf("%s\n", all_ok ? "windows match one-token steps" : "MISMATCH");
    return all_ok ? 0 : 1;
}
