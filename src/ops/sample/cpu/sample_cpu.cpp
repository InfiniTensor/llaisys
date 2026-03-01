#include "sample_cpu.hpp"
#include "../../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace llaisys::ops::cpu {
namespace {
template <typename T>
void sample_t(
    std::byte *out_idx,
    const std::byte *logits,
    size_t numel,
    int top_k,
    float top_p,
    float temperature,
    int64_t seed) {
    const T *in = reinterpret_cast<const T *>(logits);
    std::vector<float> scores(numel);
    for (size_t i = 0; i < numel; ++i) {
        scores[i] = llaisys::utils::cast<float>(in[i]) / temperature;
    }

    size_t k = std::min(static_cast<size_t>(top_k), numel);
    std::vector<size_t> order(numel);
    std::iota(order.begin(), order.end(), 0);
    if (k < numel) {
        std::partial_sort(
            order.begin(),
            order.begin() + static_cast<ptrdiff_t>(k),
            order.end(),
            [&](size_t a, size_t b) { return scores[a] > scores[b]; });

        std::vector<char> keep(numel, 0);
        for (size_t i = 0; i < k; ++i) {
            keep[order[i]] = 1;
        }

        const float neg_inf = -std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < numel; ++i) {
            if (!keep[i]) {
                scores[i] = neg_inf;
            }
        }
    }

    float max_score = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < numel; ++i) {
        if (std::isfinite(scores[i]) && scores[i] > max_score) {
            max_score = scores[i];
        }
    }
    ASSERT(std::isfinite(max_score), "sample: invalid scores after top_k filter");

    std::vector<float> probs(numel, 0.0f);
    float sum = 0.0f;
    for (size_t i = 0; i < numel; ++i) {
        if (!std::isfinite(scores[i])) {
            continue;
        }
        probs[i] = std::exp(scores[i] - max_score);
        sum += probs[i];
    }
    ASSERT(sum > 0.0f, "sample: invalid probability sum after softmax");
    for (size_t i = 0; i < numel; ++i) {
        probs[i] /= sum;
    }

    if (top_p < 1.0f) {
        std::vector<size_t> p_order(numel);
        std::iota(p_order.begin(), p_order.end(), 0);
        std::sort(
            p_order.begin(),
            p_order.end(),
            [&](size_t a, size_t b) { return probs[a] > probs[b]; });

        std::vector<char> keep(numel, 0);
        float cum = 0.0f;
        for (size_t idx : p_order) {
            if (probs[idx] <= 0.0f) {
                continue;
            }
            keep[idx] = 1;
            cum += probs[idx];
            if (cum >= top_p) {
                break;
            }
        }
        if (cum <= 0.0f && !p_order.empty()) {
            keep[p_order[0]] = 1;
        }

        float keep_sum = 0.0f;
        for (size_t i = 0; i < numel; ++i) {
            if (!keep[i]) {
                probs[i] = 0.0f;
                continue;
            }
            keep_sum += probs[i];
        }
        ASSERT(keep_sum > 0.0f, "sample: invalid probability sum after top_p filter");
        for (size_t i = 0; i < numel; ++i) {
            probs[i] /= keep_sum;
        }
    }

    std::mt19937_64 rng;
    if (seed >= 0) {
        rng.seed(static_cast<uint64_t>(seed));
    } else {
        std::random_device rd;
        rng.seed(static_cast<uint64_t>(rd()));
    }

    std::discrete_distribution<size_t> dist(probs.begin(), probs.end());
    reinterpret_cast<int64_t *>(out_idx)[0] = static_cast<int64_t>(dist(rng));
}
} // namespace

void sample(
    std::byte *out_idx,
    const std::byte *logits,
    llaisysDataType_t dtype,
    size_t numel,
    int top_k,
    float top_p,
    float temperature,
    int64_t seed) {
    ASSERT(out_idx != nullptr, "sample: out_idx is null");
    ASSERT(logits != nullptr, "sample: logits is null");

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return sample_t<float>(out_idx, logits, numel, top_k, top_p, temperature, seed);
    case LLAISYS_DTYPE_F16:
        return sample_t<llaisys::fp16_t>(out_idx, logits, numel, top_k, top_p, temperature, seed);
    case LLAISYS_DTYPE_BF16:
        return sample_t<llaisys::bf16_t>(out_idx, logits, numel, top_k, top_p, temperature, seed);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu
