#include "bl/cpu_q2.h"

#include <immintrin.h>

#include <cmath>
#include <cstring>

namespace bl::cpu {

namespace {

inline float fp16_to_f32(uint16_t h) { return _cvtsh_ss(h); }

}  // namespace

size_t q8planes_bytes(int K) { return static_cast<size_t>(K / 64) * sizeof(Q8Planes::Block); }

void q8planes_quantize(const float * x, int K, void * out) {
    auto * b = static_cast<Q8Planes::Block *>(out);
    for (int blk = 0; blk < K / 64; ++blk, x += 64, ++b) {
        for (int h = 0; h < 2; ++h) {   // like ggml's quantize_row_q8_0: d = amax/127, q = round(x/d)
            float amax = 0.f;
            for (int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(x[32 * h + i]));
            const float d = amax / 127.f, id = d ? 1.f / d : 0.f;
            int sum = 0;
            for (int i = 0; i < 32; ++i) {
                const int e = 32 * h + i;   // value index in the block
                const int q = static_cast<int>(std::nearbyint(x[e] * id));
                b->q[(e & 3) * 16 + (e >> 2)] = static_cast<int8_t>(q);
                sum += q;
            }
            b->d[h] = d;
            b->dsum[h] = d * static_cast<float>(sum);
        }
    }
}

float q2_0_dot(const uint8_t * row, const void * q8p, int K) {
    const auto * b = static_cast<const Q8Planes::Block *>(q8p);
    const __m256i m3 = _mm256_set1_epi8(3), ones = _mm256_set1_epi16(1);
    // int32 lane i of a plane pair: lanes 0,1 and 4,5 hold k = 0..7 (values < 32), lanes 2,3 and 6,7 k = 8..15
    __m256 acc = _mm256_setzero_ps();
    float corr = 0.f;
    for (int blk = 0; blk < K / 64; ++blk, row += 18, ++b) {
        uint16_t dh;
        std::memcpy(&dh, row, 2);
        const float d2 = fp16_to_f32(dh);
        const __m128i qs = _mm_loadu_si128(reinterpret_cast<const __m128i *>(row + 2));
        const __m256i p01 = _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(qs, 2), qs), m3);
        const __m256i p23 = _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(qs, 6), _mm_srli_epi16(qs, 4)), m3);
        const __m256i x01 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b->q));
        const __m256i x23 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b->q + 32));
        const __m256i s = _mm256_add_epi32(_mm256_madd_epi16(_mm256_maddubs_epi16(p01, x01), ones),
                                           _mm256_madd_epi16(_mm256_maddubs_epi16(p23, x23), ones));
        const __m256 scale = _mm256_mul_ps(_mm256_set1_ps(d2), _mm256_setr_ps(b->d[0], b->d[0], b->d[1], b->d[1],
                                                                               b->d[0], b->d[0], b->d[1], b->d[1]));
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), scale, acc);
        corr += d2 * (b->dsum[0] + b->dsum[1]);
    }
    const __m128 h = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    const __m128 h2 = _mm_add_ps(h, _mm_movehl_ps(h, h));
    return _mm_cvtss_f32(_mm_add_ss(h2, _mm_shuffle_ps(h2, h2, 1))) - corr;
}

}  // namespace bl::cpu
