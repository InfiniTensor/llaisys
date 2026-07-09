#include "random_sample_cpu.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace llaisys::ops::cpu {

template <typename T>
void read_logits_to_float(
    std::vector<float> &dst,
    const std::byte *src,
    size_t numel,
    float inv_temperature) {
    const T *typed_src = reinterpret_cast<const T *>(src);
    dst.resize(numel);
    for (size_t i = 0; i < numel; ++i) {
        float v = llaisys::utils::cast<float>(typed_src[i]);
        if (!std::isfinite(v)) {
            v = v > 0.0f ? 1e30f : -1e30f;
        }
        dst[i] = v * inv_temperature;
    }
}

static int64_t argmax_index(const std::vector<float> &logits) {
    int64_t best_idx = 0;
    float best_val = logits[0];
    for (size_t i = 1; i < logits.size(); ++i) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best_idx = static_cast<int64_t>(i);
        }
    }
    return best_idx;
}

static int64_t sample_index_from_logits(
    const std::vector<float> &scaled_logits,
    size_t top_k,
    float top_p,
    uint64_t seed) {
    const size_t n = scaled_logits.size();
    if (n == 0) {
        return 0;
    }

    if (top_k == 0 || top_k > n) {
        top_k = n;
    }
    if (!std::isfinite(top_p)) {
        top_p = 1.0f;
    }
    if (top_p < 0.0f) {
        top_p = 0.0f;
    }
    if (top_p > 1.0f) {
        top_p = 1.0f;
    }

    std::vector<size_t> candidates(n);
    std::iota(candidates.begin(), candidates.end(), size_t(0));

    auto by_logit_desc = [&scaled_logits](size_t lhs, size_t rhs) {
        const float lv = scaled_logits[lhs];
        const float rv = scaled_logits[rhs];
        if (lv == rv) {
            return lhs < rhs;
        }
        return lv > rv;
    };

    if (top_k < n) {
        std::nth_element(
            candidates.begin(),
            candidates.begin() + static_cast<ptrdiff_t>(top_k),
            candidates.end(),
            by_logit_desc);
        candidates.resize(top_k);
    }

    std::sort(candidates.begin(), candidates.end(), by_logit_desc);

    if (top_p <= 0.0f || candidates.size() == 1) {
        return static_cast<int64_t>(candidates[0]);
    }

    if (top_p < 1.0f) {
        float max_logit = scaled_logits[candidates[0]];
        std::vector<double> probs(candidates.size(), 0.0);
        double prob_sum = 0.0;
        for (size_t i = 0; i < candidates.size(); ++i) {
            const double w = std::exp(static_cast<double>(scaled_logits[candidates[i]] - max_logit));
            probs[i] = w;
            prob_sum += w;
        }

        if (!(prob_sum > 0.0) || !std::isfinite(prob_sum)) {
            return static_cast<int64_t>(candidates[0]);
        }

        double cumulative = 0.0;
        size_t keep = 0;
        for (size_t i = 0; i < candidates.size(); ++i) {
            cumulative += probs[i] / prob_sum;
            keep = i + 1;
            if (cumulative >= static_cast<double>(top_p)) {
                break;
            }
        }
        if (keep == 0) {
            keep = 1;
        }
        candidates.resize(keep);
    }

    float max_logit = -std::numeric_limits<float>::infinity();
    for (size_t idx : candidates) {
        if (scaled_logits[idx] > max_logit) {
            max_logit = scaled_logits[idx];
        }
    }

    std::vector<double> weights(candidates.size(), 0.0);
    double weight_sum = 0.0;
    for (size_t i = 0; i < candidates.size(); ++i) {
        const double w = std::exp(static_cast<double>(scaled_logits[candidates[i]] - max_logit));
        weights[i] = w;
        weight_sum += w;
    }

    if (!(weight_sum > 0.0) || !std::isfinite(weight_sum)) {
        return static_cast<int64_t>(candidates[0]);
    }

    std::mt19937_64 rng(seed);
    std::discrete_distribution<size_t> dist(weights.begin(), weights.end());
    return static_cast<int64_t>(candidates[dist(rng)]);
}

void random_sample(
    std::byte *out_idx,
    const std::byte *logits,
    llaisysDataType_t type,
    size_t numel,
    float temperature,
    size_t top_k,
    float top_p,
    uint64_t seed) {
    ASSERT(out_idx != nullptr, "RandomSample: out_idx cannot be null.");
    ASSERT(logits != nullptr, "RandomSample: logits cannot be null.");
    ASSERT(numel > 0, "RandomSample: numel must be positive.");

    std::vector<float> scaled_logits;
    bool deterministic = !std::isfinite(temperature) || temperature <= 0.0f || top_k == 1 || top_p <= 0.0f;
    const float inv_temperature = deterministic ? 1.0f : (1.0f / temperature);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        read_logits_to_float<float>(scaled_logits, logits, numel, inv_temperature);
        break;
    case LLAISYS_DTYPE_F16:
        read_logits_to_float<llaisys::fp16_t>(scaled_logits, logits, numel, inv_temperature);
        break;
    case LLAISYS_DTYPE_BF16:
        read_logits_to_float<llaisys::bf16_t>(scaled_logits, logits, numel, inv_temperature);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    int64_t sampled = 0;
    if (deterministic) {
        sampled = argmax_index(scaled_logits);
    } else {
        sampled = sample_index_from_logits(scaled_logits, top_k, top_p, seed);
    }
    *reinterpret_cast<int64_t *>(out_idx) = sampled;
}

} // namespace llaisys::ops::cpu
