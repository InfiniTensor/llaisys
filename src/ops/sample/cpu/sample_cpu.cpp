#include "sample_cpu.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace {

template <typename T>
int64_t argmax_index(const T *vals, size_t numel) {
    size_t max_index = 0;
    float max_value = llaisys::utils::cast<float>(vals[0]);

    for (size_t i = 1; i < numel; ++i) {
        const float current = llaisys::utils::cast<float>(vals[i]);
        if (current > max_value) {
            max_value = current;
            max_index = i;
        }
    }

    return static_cast<int64_t>(max_index);
}

template <typename T>
int64_t sample_impl(const T *vals, size_t numel, int top_k, float top_p, float temperature) {
    if (numel == 0) {
        return 0;
    }

    if (!std::isfinite(temperature) || temperature <= 0.0f || top_k == 1) {
        return argmax_index(vals, numel);
    }

    struct Candidate {
        size_t index;
        float logit;
        float weight;
    };

    const size_t k = top_k <= 0 ? numel : std::min(numel, static_cast<size_t>(top_k));
    const float safe_top_p = (!std::isfinite(top_p) || top_p <= 0.0f || top_p > 1.0f) ? 1.0f : top_p;
    const float inv_temperature = 1.0f / temperature;

    std::vector<Candidate> candidates;
    candidates.reserve(numel);
    for (size_t i = 0; i < numel; ++i) {
        candidates.push_back({i, llaisys::utils::cast<float>(vals[i]) * inv_temperature, 0.0f});
    }

    auto by_logit_desc = [](const Candidate &lhs, const Candidate &rhs) {
        return lhs.logit > rhs.logit;
    };
    std::partial_sort(candidates.begin(), candidates.begin() + k, candidates.end(), by_logit_desc);
    candidates.resize(k);

    const float max_logit = candidates.front().logit;
    float total_mass = 0.0f;
    for (auto &candidate : candidates) {
        candidate.weight = std::exp(candidate.logit - max_logit);
        total_mass += candidate.weight;
    }

    if (safe_top_p < 1.0f && total_mass > 0.0f) {
        float cumulative = 0.0f;
        size_t keep = 0;
        for (; keep < candidates.size(); ++keep) {
            cumulative += candidates[keep].weight / total_mass;
            if (cumulative >= safe_top_p) {
                ++keep;
                break;
            }
        }
        candidates.resize(std::max<size_t>(1, std::min(keep, candidates.size())));
    }

    float kept_mass = 0.0f;
    for (const auto &candidate : candidates) {
        kept_mass += candidate.weight;
    }
    if (!(kept_mass > 0.0f)) {
        return static_cast<int64_t>(candidates.front().index);
    }

    thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, kept_mass);
    const float draw = dist(rng);

    float running = 0.0f;
    for (const auto &candidate : candidates) {
        running += candidate.weight;
        if (draw <= running) {
            return static_cast<int64_t>(candidate.index);
        }
    }

    return static_cast<int64_t>(candidates.back().index);
}

} // namespace

namespace llaisys::ops::cpu {
int64_t sample(const std::byte *logits, llaisysDataType_t type, size_t numel, int top_k, float top_p, float temperature) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return sample_impl(reinterpret_cast<const float *>(logits), numel, top_k, top_p, temperature);
    case LLAISYS_DTYPE_BF16:
        return sample_impl(reinterpret_cast<const llaisys::bf16_t *>(logits), numel, top_k, top_p, temperature);
    case LLAISYS_DTYPE_F16:
        return sample_impl(reinterpret_cast<const llaisys::fp16_t *>(logits), numel, top_k, top_p, temperature);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
