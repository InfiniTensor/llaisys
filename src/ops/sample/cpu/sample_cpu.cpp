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
float to_float(T value) {
    if constexpr (std::is_same_v<T, llaisys::fp16_t> || std::is_same_v<T, llaisys::bf16_t>) {
        return llaisys::utils::cast<float>(value);
    } else {
        return static_cast<float>(value);
    }
}

template <typename T>
int64_t sample_impl(const std::byte *logits, size_t numel, float temperature, int top_k, float top_p, uint64_t seed) {
    const auto *typed_logits = reinterpret_cast<const T *>(logits);
    std::vector<int64_t> indices(numel);
    std::iota(indices.begin(), indices.end(), 0);

    std::vector<float> scaled_logits(numel);
    float inv_temperature = temperature > 0.0f ? (1.0f / temperature) : 1.0f;
    for (size_t i = 0; i < numel; ++i) {
        scaled_logits[i] = to_float(typed_logits[i]) * inv_temperature;
    }

    std::sort(indices.begin(), indices.end(), [&](int64_t lhs, int64_t rhs) {
        return scaled_logits[lhs] > scaled_logits[rhs];
    });

    if (top_k > 0 && static_cast<size_t>(top_k) < indices.size()) {
        indices.resize(static_cast<size_t>(top_k));
    }

    float max_logit = -std::numeric_limits<float>::infinity();
    for (int64_t idx : indices) {
        max_logit = std::max(max_logit, scaled_logits[idx]);
    }

    std::vector<float> probs(indices.size());
    float prob_sum = 0.0f;
    for (size_t i = 0; i < indices.size(); ++i) {
        probs[i] = std::exp(scaled_logits[indices[i]] - max_logit);
        prob_sum += probs[i];
    }

    for (float &prob : probs) {
        prob /= prob_sum;
    }

    if (top_p > 0.0f && top_p < 1.0f) {
        float cumulative = 0.0f;
        size_t keep = 0;
        for (; keep < probs.size(); ++keep) {
            cumulative += probs[keep];
            if (cumulative >= top_p) {
                ++keep;
                break;
            }
        }
        keep = std::max<size_t>(keep, 1);
        indices.resize(keep);
        probs.resize(keep);
        float new_sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
        for (float &prob : probs) {
            prob /= new_sum;
        }
    }

    std::mt19937_64 rng(seed);
    std::discrete_distribution<size_t> dist(probs.begin(), probs.end());
    return indices[dist(rng)];
}
} // namespace

int64_t sample(const std::byte *logits, size_t numel, llaisysDataType_t dtype, float temperature, int top_k, float top_p, uint64_t seed) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return sample_impl<float>(logits, numel, temperature, top_k, top_p, seed);
    case LLAISYS_DTYPE_F16:
        return sample_impl<llaisys::fp16_t>(logits, numel, temperature, top_k, top_p, seed);
    case LLAISYS_DTYPE_BF16:
        return sample_impl<llaisys::bf16_t>(logits, numel, temperature, top_k, top_p, seed);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
