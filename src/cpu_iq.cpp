#include "bl/cpu_iq.h"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>

#include <cstring>

namespace bl::cpu {

namespace {

inline float h2f(uint16_t h) { return _cvtsh_ss(h); }
inline uint32_t ld32(const uint8_t * p) { uint32_t v; std::memcpy(&v, p, 4); return v; }

// 8-bit sign masks -> 8 bytes of -1 (bit set) / +1, for _mm256_sign_epi8
struct SignTables {
    uint64_t byte[256];   // any 8-bit mask
    uint64_t even[128];   // the 7-bit index of ksigns_iq2xs (the 8th sign is the parity)
    SignTables() {
        for (int m = 0; m < 256; ++m) {
            uint64_t r = 0;
            for (int k = 0; k < 8; ++k) r |= static_cast<uint64_t>(((m >> k) & 1) ? 0xFF : 0x01) << (8 * k);
            byte[m] = r;
        }
        for (int i = 0; i < 128; ++i) even[i] = byte[ksigns_iq2xs[i]];
    }
};
const SignTables sgn;

inline __m256i set4(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {   // a: bytes 0-7 ... d: bytes 24-31
    return _mm256_set_epi64x(static_cast<long long>(d), static_cast<long long>(c), static_cast<long long>(b),
                             static_cast<long long>(a));
}
inline uint64_t grid4x2(const uint32_t * grid, int i0, int i1) {   // two 4-value grid entries as 8 bytes
    return static_cast<uint64_t>(grid[i0]) | static_cast<uint64_t>(grid[i1]) << 32;
}
inline __m256i scale2(int a, int b) {   // int16 lanes: a for values 0-15, b for 16-31 of the chunk
    return _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_set1_epi16(static_cast<short>(a))), _mm_set1_epi16(static_cast<short>(b)), 1);
}

// Per format: chunk ib (32 values) of a 256-value block -> magnitudes (u8), signs (+-1), int16 scales; value =
// K * d * scale * magnitude * sign
template <int F> struct Iq;

template <> struct Iq<16> {   // IQ2_XXS
    using B = block_iq2_xxs;
    static constexpr float K = 0.125f;
    static void chunk(const B & b, int ib, __m256i & mag, __m256i & s, __m256i & sc) {
        const uint8_t * q = reinterpret_cast<const uint8_t *>(b.qs) + 8 * ib;
        const uint32_t a = ld32(q), w = ld32(q + 4);
        mag = set4(iq2xxs_grid[a & 255], iq2xxs_grid[(a >> 8) & 255], iq2xxs_grid[(a >> 16) & 255], iq2xxs_grid[a >> 24]);
        s = set4(sgn.even[w & 127], sgn.even[(w >> 7) & 127], sgn.even[(w >> 14) & 127], sgn.even[(w >> 21) & 127]);
        sc = _mm256_set1_epi16(static_cast<short>(2 * (w >> 28) + 1));
    }
};
template <> struct Iq<17> {   // IQ2_XS
    using B = block_iq2_xs;
    static constexpr float K = 0.125f;
    static void chunk(const B & b, int ib, __m256i & mag, __m256i & s, __m256i & sc) {
        const uint16_t * q = b.qs + 4 * ib;
        mag = set4(iq2xs_grid[q[0] & 511], iq2xs_grid[q[1] & 511], iq2xs_grid[q[2] & 511], iq2xs_grid[q[3] & 511]);
        s = set4(sgn.even[q[0] >> 9], sgn.even[q[1] >> 9], sgn.even[q[2] >> 9], sgn.even[q[3] >> 9]);
        sc = scale2(2 * (b.scales[ib] & 15) + 1, 2 * (b.scales[ib] >> 4) + 1);
    }
};
template <> struct Iq<22> {   // IQ2_S
    using B = block_iq2_s;
    static constexpr float K = 0.125f;
    static void chunk(const B & b, int ib, __m256i & mag, __m256i & s, __m256i & sc) {
        const uint8_t * q = b.qs + 4 * ib, * sg = b.qs + 32 + 4 * ib;
        const int h = b.qh[ib];
        mag = set4(iq2s_grid[q[0] | ((h << 8) & 0x300)], iq2s_grid[q[1] | ((h << 6) & 0x300)],
                   iq2s_grid[q[2] | ((h << 4) & 0x300)], iq2s_grid[q[3] | ((h << 2) & 0x300)]);
        s = set4(sgn.byte[sg[0]], sgn.byte[sg[1]], sgn.byte[sg[2]], sgn.byte[sg[3]]);
        sc = scale2(2 * (b.scales[ib] & 15) + 1, 2 * (b.scales[ib] >> 4) + 1);
    }
};
template <> struct Iq<18> {   // IQ3_XXS
    using B = block_iq3_xxs;
    static constexpr float K = 0.25f;
    static void chunk(const B & b, int ib, __m256i & mag, __m256i & s, __m256i & sc) {
        const uint8_t * q = b.qs + 8 * ib;
        const uint32_t w = ld32(b.qs + 64 + 4 * ib);
        mag = set4(grid4x2(iq3xxs_grid, q[0], q[1]), grid4x2(iq3xxs_grid, q[2], q[3]), grid4x2(iq3xxs_grid, q[4], q[5]),
                   grid4x2(iq3xxs_grid, q[6], q[7]));
        s = set4(sgn.even[w & 127], sgn.even[(w >> 7) & 127], sgn.even[(w >> 14) & 127], sgn.even[(w >> 21) & 127]);
        sc = _mm256_set1_epi16(static_cast<short>(2 * (w >> 28) + 1));
    }
};
template <> struct Iq<21> {   // IQ3_S
    using B = block_iq3_s;
    static constexpr float K = 1.f;
    static void chunk(const B & b, int ib, __m256i & mag, __m256i & s, __m256i & sc) {
        const uint8_t * q = b.qs + 8 * ib, * sg = b.signs + 4 * ib;
        const int h = b.qh[ib];
        auto gi = [&](int l) { return q[l] | ((h << (8 - l)) & 256); };
        mag = set4(grid4x2(iq3s_grid, gi(0), gi(1)), grid4x2(iq3s_grid, gi(2), gi(3)), grid4x2(iq3s_grid, gi(4), gi(5)),
                   grid4x2(iq3s_grid, gi(6), gi(7)));
        s = set4(sgn.byte[sg[0]], sgn.byte[sg[1]], sgn.byte[sg[2]], sgn.byte[sg[3]]);
        sc = _mm256_set1_epi16(static_cast<short>(1 + 2 * ((b.scales[ib / 2] >> (4 * (ib & 1))) & 15)));
    }
};

inline float hsum(__m256 v) {
    __m128 x = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    x = _mm_add_ps(x, _mm_movehl_ps(x, x));
    x = _mm_add_ss(x, _mm_movehdup_ps(x));
    return _mm_cvtss_f32(x);
}

template <int F, int NT>
void gate_up(const uint8_t * gr, const uint8_t * ur, int K, const void * const * x, float * g, float * u) {
    using T = Iq<F>;
    const auto * gb = reinterpret_cast<const typename T::B *>(gr);
    const auto * ub = reinterpret_cast<const typename T::B *>(ur);
    const block_q8_K * xb[NT];
    for (int t = 0; t < NT; ++t) xb[t] = static_cast<const block_q8_K *>(x[t]);
    __m256 fg[NT], fu[NT];
    for (int t = 0; t < NT; ++t) { fg[t] = _mm256_setzero_ps(); fu[t] = _mm256_setzero_ps(); }
    for (int i = 0; i < K / QK_K; ++i) {
        _mm_prefetch(reinterpret_cast<const char *>(gb + i + 2), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char *>(ub + i + 2), _MM_HINT_T0);
        __m256i ag[NT], au[NT];
        for (int t = 0; t < NT; ++t) { ag[t] = _mm256_setzero_si256(); au[t] = _mm256_setzero_si256(); }
        for (int ib = 0; ib < QK_K / 32; ++ib) {
            __m256i gm, gs, gsc, um, us, usc;
            T::chunk(gb[i], ib, gm, gs, gsc);
            T::chunk(ub[i], ib, um, us, usc);
            for (int t = 0; t < NT; ++t) {
                const __m256i q = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(xb[t][i].qs + 32 * ib));
                ag[t] = _mm256_add_epi32(ag[t], _mm256_madd_epi16(_mm256_maddubs_epi16(gm, _mm256_sign_epi8(q, gs)), gsc));
                au[t] = _mm256_add_epi32(au[t], _mm256_madd_epi16(_mm256_maddubs_epi16(um, _mm256_sign_epi8(q, us)), usc));
            }
        }
        const float dg = h2f(gb[i].d), du = h2f(ub[i].d);
        for (int t = 0; t < NT; ++t) {
            const float dx = xb[t][i].d;
            fg[t] = _mm256_fmadd_ps(_mm256_set1_ps(dg * dx), _mm256_cvtepi32_ps(ag[t]), fg[t]);
            fu[t] = _mm256_fmadd_ps(_mm256_set1_ps(du * dx), _mm256_cvtepi32_ps(au[t]), fu[t]);
        }
    }
    for (int t = 0; t < NT; ++t) { g[t] = T::K * hsum(fg[t]); u[t] = T::K * hsum(fu[t]); }
}

template <int F>
void gate_up_n(const uint8_t * gr, const uint8_t * ur, int K, const void * const * x, int nt, float * g, float * u) {
    switch (nt) {
        case 1: return gate_up<F, 1>(gr, ur, K, x, g, u);
        case 2: return gate_up<F, 2>(gr, ur, K, x, g, u);
        case 3: return gate_up<F, 3>(gr, ur, K, x, g, u);
        case 4: return gate_up<F, 4>(gr, ur, K, x, g, u);
        default:   // more tokens: in groups of up to 4
            for (int t0 = 0; t0 < nt; t0 += 4) gate_up_n<F>(gr, ur, K, x + t0, nt - t0 < 4 ? nt - t0 : 4, g + t0, u + t0);
    }
}

}  // namespace

bool iq_supported(uint32_t t) { return t == 16 || t == 17 || t == 18 || t == 21 || t == 22; }

void iq_gate_up(uint32_t type, const uint8_t * gr, const uint8_t * ur, int K, const void * const * x, int nt, float * g, float * u) {
    switch (type) {
        case 16: return gate_up_n<16>(gr, ur, K, x, nt, g, u);
        case 17: return gate_up_n<17>(gr, ur, K, x, nt, g, u);
        case 18: return gate_up_n<18>(gr, ur, K, x, nt, g, u);
        case 21: return gate_up_n<21>(gr, ur, K, x, nt, g, u);
        case 22: return gate_up_n<22>(gr, ur, K, x, nt, g, u);
        default: break;
    }
}

}  // namespace bl::cpu
