#include "random_sample_cpu.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

template <typename T>
void random_sample_(int64_t *sample_idx, T *sample_val, const T *logits, size_t vocab_size, float temperature,
                    int top_k, float top_p, int64_t seed) {
    if (top_p <= 0.0f) {
        top_k = 1;
    }

    std::vector<float> scaled_logits(vocab_size);
    float inv_temperature = 1.0f / std::max(temperature, 1e-6f);

    for (size_t i = 0; i < vocab_size; ++i) {
        scaled_logits[i] = llaisys::utils::cast<float>(logits[i]) * inv_temperature;
    }

    std::vector<size_t> sorted_indices(vocab_size);
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(), sorted_indices.end(),
              [&](size_t a, size_t b) { return scaled_logits[a] > scaled_logits[b]; });

    if (top_k <= 0 || static_cast<size_t>(top_k) > vocab_size) {
        top_k = static_cast<int>(vocab_size);
    }
    sorted_indices.resize(static_cast<size_t>(top_k));

    std::vector<float> candidate_probs(sorted_indices.size());
    float max_logit = scaled_logits[sorted_indices[0]];
    float sum_exp = 0.0f;
    for (size_t i = 0; i < sorted_indices.size(); ++i) {
        float p = std::exp(scaled_logits[sorted_indices[i]] - max_logit);
        candidate_probs[i] = p;
        sum_exp += p;
    }
    for (float &p : candidate_probs) {
        p /= sum_exp;
    }

    float top_p_clamped = std::min(1.0f, std::max(0.0f, top_p));
    size_t keep_count = sorted_indices.size();
    if (top_p_clamped > 0.0f && top_p_clamped < 1.0f) {
        float cumulative = 0.0f;
        keep_count = 0;
        for (size_t i = 0; i < candidate_probs.size(); ++i) {
            cumulative += candidate_probs[i];
            keep_count = i + 1;
            if (cumulative >= top_p_clamped) {
                break;
            }
        }
        keep_count = std::max<size_t>(1, keep_count);
    }

    sorted_indices.resize(keep_count);
    candidate_probs.resize(keep_count);

    float renorm_sum = 0.0f;
    for (float p : candidate_probs) {
        renorm_sum += p;
    }
    for (float &p : candidate_probs) {
        p /= renorm_sum;
    }

    std::discrete_distribution<size_t> distribution(candidate_probs.begin(), candidate_probs.end());
    size_t sampled_local_idx = 0;
    if (seed >= 0) {
        std::mt19937_64 generator(static_cast<uint64_t>(seed));
        sampled_local_idx = distribution(generator);
    } else {
        static thread_local std::mt19937_64 generator(std::random_device{}());
        sampled_local_idx = distribution(generator);
    }
    size_t sampled_vocab_idx = sorted_indices[sampled_local_idx];

    sample_idx[0] = static_cast<int64_t>(sampled_vocab_idx);
    sample_val[0] = logits[sampled_vocab_idx];
}

namespace llaisys::ops::cpu {
void random_sample(std::byte *sample_idx, std::byte *sample_val, const std::byte *logits, llaisysDataType_t type,
                   size_t vocab_size, float temperature, int top_k, float top_p, int64_t seed) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return random_sample_(reinterpret_cast<int64_t *>(sample_idx), reinterpret_cast<float *>(sample_val),
                              reinterpret_cast<const float *>(logits), vocab_size, temperature, top_k, top_p, seed);
    case LLAISYS_DTYPE_BF16:
        return random_sample_(reinterpret_cast<int64_t *>(sample_idx),
                              reinterpret_cast<llaisys::bf16_t *>(sample_val),
                              reinterpret_cast<const llaisys::bf16_t *>(logits), vocab_size, temperature, top_k,
                              top_p, seed);
    case LLAISYS_DTYPE_F16:
        return random_sample_(reinterpret_cast<int64_t *>(sample_idx),
                              reinterpret_cast<llaisys::fp16_t *>(sample_val),
                              reinterpret_cast<const llaisys::fp16_t *>(logits), vocab_size, temperature, top_k,
                              top_p, seed);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
