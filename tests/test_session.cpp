// The generate() session invariant: after a call the state is everything fed and emitted except the last emitted
// token (pending), so chained turns give exactly what one call over the concatenated text gives.
//   test_session SHARD1.gguf PROMPT1.tokens PROMPT2.tokens LONG.tokens
// Exact comparisons need run-independent numerics: run with BL_CPU_SHARE=0 BL_NO_Q8=1 BL_PREFILL=0 (the window path,
// bit-identical however the tokens are grouped into windows).
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

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

std::vector<int> cat(std::vector<int> a, const std::vector<int> & b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

int fails = 0;
void check(bool ok, const std::string & what) {
    std::printf("  %-72s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    fails += !ok;
}
// equal outputs; on a mismatch, how far they agree (in the default, non-deterministic mode a late divergence is the
// numerics of different window groupings, an early one a state error)
void check_same(const std::vector<int> & a, const std::vector<int> & b, const std::string & what) {
    size_t n = 0;
    while (n < a.size() && n < b.size() && a[n] == b[n]) ++n;
    const bool ok = a == b;
    check(ok, ok ? what : what + " (agree on " + std::to_string(n) + " of " + std::to_string(a.size()) + ")");
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 5) { std::fprintf(stderr, "usage: %s SHARD1.gguf P1 P2 LONG\n", argv[0]); return 2; }
    std::vector<int> p1 = read_tokens(argv[2]), p2 = read_tokens(argv[3]), lg = read_tokens(argv[4]);
    if (p1.size() > 40) p1.resize(40);
    if (p2.size() > 30) p2.resize(30);
    constexpr int kCtx = 512;
    bl::Engine eng(argv[1], kCtx);
    const int n1 = 25, n2 = 25;

    for (int spec : {3, 0}) {
        std::printf("spec %d\n", spec);
        // 1. two turns vs one call over [p1, out1, p2]
        eng.reset();
        const std::vector<int> out1 = eng.generate(p1, n1, spec);
        check(eng.pending() == out1.back(), "after turn 1: the last emitted token is pending");
        check(eng.position() == static_cast<int>(p1.size() + out1.size()) - 1, "after turn 1: position = fed + emitted - 1");
        const std::vector<int> out2 = eng.generate(p2, n2, spec);
        eng.reset();
        const std::vector<int> ref2 = eng.generate(cat(cat(p1, out1), p2), n2, spec);
        check_same(out2, ref2, "turn 2 equals one call over [p1, out1, p2]");
        {   // control: the old behaviour (turn 1's last token never fed) - how early it departs from the reference
            eng.reset();
            std::vector<int> o1 = out1;
            o1.pop_back();
            const std::vector<int> bug = eng.generate(cat(cat(p1, o1), p2), n2, spec);
            size_t n = 0;
            while (n < bug.size() && bug[n] == ref2[n]) ++n;
            std::printf("  (control: dropping turn 1's last token agrees with the reference on %zu of %zu)\n", n, ref2.size());
        }

        // 2. a stop mid-window (after the 7th token), then a second turn
        eng.reset();
        int seen = 0;
        const std::vector<int> s1 = eng.generate(p1, 100, spec, 0.5f, [&](int) { return ++seen < 7; });
        check(s1.size() == 7 && eng.pending() == s1.back(), "callback stop: 7 emitted, the 7th pending");
        check(eng.position() == static_cast<int>(p1.size()) + 6, "callback stop: nothing past the stop committed");
        const std::vector<int> s2 = eng.generate(p2, n2, spec);
        eng.reset();
        const std::vector<int> sref = eng.generate(cat(cat(p1, s1), p2), n2, spec);
        check_same(s2, sref, "after a callback stop: turn 2 equals one call");

        // 3. max_new, then continuing with an empty prompt
        eng.reset();
        const std::vector<int> a = eng.generate(p1, 5, spec);
        const std::vector<int> b = eng.generate({}, 20, spec);
        eng.reset();
        const std::vector<int> c = eng.generate(p1, 25, spec);
        check(a.size() == 5, "max_new: 5 emitted");
        check_same(cat(a, b), c, "generate(p1, 5) + generate({}, 20) equals generate(p1, 25)");
    }

    // 4. a full context stops the generation instead of throwing
    {
        std::printf("context\n");
        eng.reset();
        std::vector<int> big(lg.begin(), lg.begin() + std::min<size_t>(lg.size(), kCtx - 20));
        bool threw = false;
        std::vector<int> out;
        try { out = eng.generate(big, 100, 3); } catch (const std::exception & e) { threw = true; std::printf("  threw: %s\n", e.what()); }
        check(!threw, "no exception when the context fills");
        check(eng.position() <= kCtx && eng.position() + 1 >= kCtx, "stopped with the context full");
        check(static_cast<int>(big.size() + out.size()) - 1 == eng.position(), "state = fed + emitted - 1 at the limit");
    }
    std::printf("%s\n", fails ? "SESSION MISMATCH" : "session invariant holds");
    return fails ? 1 : 0;
}
