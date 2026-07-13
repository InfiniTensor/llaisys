#include "sample_cpu.hpp"
#include "../../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

// Thread-local RNG so each thread has independent, seeding-capable state.
static thread_local std::mt19937_64 rng{std::random_device{}()};

namespace llaisys::ops::cpu {

void sample_set_seed(uint64_t seed) {
    rng.seed(seed);
}

template <typename T>
static int64_t sample_impl(const T *logits, size_t numel,
                           int top_k, float top_p, float temperature) {
    // ── 1. Temperature scaling (in float) ──────────────────────────────────
    std::vector<float> scores(numel);
    for (size_t i = 0; i < numel; ++i)
        scores[i] = casting(float, logits[i]) / temperature;

    // ── 2. Sort indices by score descending ────────────────────────────────
    std::vector<int> indices(numel);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&](int a, int b) {
                  if (scores[a] == scores[b]) {
                      // Deterministic tie-break: match argmax semantics (first index wins).
                      return a < b;
                  }
                  return scores[a] > scores[b];
              });

    // Near-zero temperature approximates greedy decoding; return argmax deterministically.
    if (temperature <= 1e-5f) {
        return static_cast<int64_t>(indices[0]);
    }

    // ── 3. Top-k truncation ────────────────────────────────────────────────
    int k = static_cast<int>(numel);
    if (top_k > 0 && top_k < k)
        k = top_k;

    // ── 4. Softmax over the top-k candidates (numerically stable) ─────────
    float max_score = scores[indices[0]];
    std::vector<float> probs(k);
    for (int i = 0; i < k; ++i)
        probs[i] = std::exp(scores[indices[i]] - max_score);

    float sum = 0.0f;
    for (int i = 0; i < k; ++i)
        sum += probs[i];
    for (int i = 0; i < k; ++i)
        probs[i] /= sum;

    // ── 5. Top-p (nucleus) truncation ─────────────────────────────────────
    // Keep the minimal prefix whose cumulative probability >= top_p, then
    // renormalise.
    if (top_p < 1.0f) {
        float cumsum = 0.0f;
        int cutoff = k;
        for (int i = 0; i < k; ++i) {
            cumsum += probs[i];
            if (cumsum >= top_p) {
                cutoff = i + 1;
                break;
            }
        }
        k = cutoff;
        probs.resize(k);
        // Renormalise after truncation.
        sum = 0.0f;
        for (int i = 0; i < k; ++i)
            sum += probs[i];
        for (int i = 0; i < k; ++i)
            probs[i] /= sum;
    }

    // ── 6. Sample from the distribution ───────────────────────────────────
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return static_cast<int64_t>(indices[dist(rng)]);
}

void sample(std::byte *out, const std::byte *logits, size_t numel,
            int top_k, float top_p, float temperature,
            llaisysDataType_t dtype) {
    int64_t result;
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        result = sample_impl(reinterpret_cast<const float *>(logits),
                             numel, top_k, top_p, temperature);
        break;
    case LLAISYS_DTYPE_F16:
        result = sample_impl(reinterpret_cast<const llaisys::fp16_t *>(logits),
                             numel, top_k, top_p, temperature);
        break;
    case LLAISYS_DTYPE_BF16:
        result = sample_impl(reinterpret_cast<const llaisys::bf16_t *>(logits),
                             numel, top_k, top_p, temperature);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
    *reinterpret_cast<int64_t *>(out) = result;
}

} // namespace llaisys::ops::cpu
