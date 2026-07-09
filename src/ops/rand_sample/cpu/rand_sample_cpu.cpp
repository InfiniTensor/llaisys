#include "rand_sample_cpu.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

template <typename T>
static void rand_sample_(std::byte *sample_idx, std::byte *sample_val, const std::byte *vals, const float temperature,
                         const size_t topK, const float topP, size_t numel, const int64_t batch_size, const int64_t seed) {
    const T *v_all = reinterpret_cast<const T *>(vals);
    auto *out_idx = reinterpret_cast<int64_t *>(sample_idx);
    auto *out_val = reinterpret_cast<T *>(sample_val);
    float temp = temperature;
    if (temp <= 1e-6f) {
        temp = 1e-6f;
    }

    std::vector<float> scores(numel);
    std::vector<std::pair<float, size_t>> sorted_scores(numel);
    std::vector<std::pair<float, size_t>> candidates;
    candidates.reserve(numel);

    std::mt19937_64 rng(static_cast<uint64_t>(seed));
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (int64_t b = 0; b < batch_size; ++b) {
        const T *v = v_all + b * numel;
        size_t max_i = 0;
        float max_v = llaisys::utils::cast<float>(v[0]);
        for (size_t i = 1; i < numel; ++i) {
            float cur = llaisys::utils::cast<float>(v[i]);
            if (cur > max_v) {
                max_v = cur;
                max_i = i;
            }
        }

        for (size_t i = 0; i < numel; ++i) {
            float cur = llaisys::utils::cast<float>(v[i]);
            scores[i] = std::exp((cur - max_v) / temp);
        }
        float sum = 0.0f;
        for (size_t i = 0; i < numel; ++i) {
            sum += scores[i];
        }
        if (sum <= 0.0f) {
            for (size_t i = 0; i < numel; ++i) {
                scores[i] = 0.0f;
            }
            scores[max_i] = 1.0f;
            sum = 1.0f;
        } else {
            for (size_t i = 0; i < numel; ++i) {
                scores[i] /= sum;
            }
        }

        for (size_t i = 0; i < numel; ++i) {
            sorted_scores[i] = {scores[i], i};
        }
        std::sort(sorted_scores.begin(), sorted_scores.end(), 
        [](const auto &a, const auto &b) { return a.first > b.first; });

        size_t k = numel;
        if (topK > 0 && topK < numel) {
            k = topK;
        }

        candidates.clear();
        if (topP > 0.0f && topP < 1.0f) {
            float cum_score = 0.0f;
            for (size_t i = 0; i < k; ++i) {
                candidates.push_back(sorted_scores[i]);
                cum_score += sorted_scores[i].first;
                if (cum_score >= topP) {
                    break;
                }
            }
        } else {
            for (size_t i = 0; i < k; ++i) {
                candidates.push_back(sorted_scores[i]);
            }
        }
        if (candidates.empty()) {
            candidates.push_back(sorted_scores[0]);
        }

        float cand_sum = 0.0f;
        for (const auto &item : candidates) {
            cand_sum += item.first;
        }

        size_t chosen = candidates[0].second;
        if (cand_sum > 0.0f) {
            float r = dist(rng);
            for (const auto &item : candidates) {
                r -= item.first / cand_sum;
                if (r <= 0.0f) {
                    chosen = item.second;
                    break;
                }
            }
            if (r > 0.0f) {
                chosen = candidates.back().second;
            }
        }

        out_idx[b] = static_cast<int64_t>(chosen);
        out_val[b] = v[chosen];
    }
}

namespace llaisys::ops::cpu {
void rand_sample(std::byte *sample_idx, std::byte *sample_val, const std::byte *vals, llaisysDataType_t type, size_t numel,
                 const int64_t batch_size, const float temperature, const size_t topK, const float topP, const int64_t seed) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rand_sample_<float>(sample_idx, sample_val, vals, 
            temperature, topK, topP, numel, batch_size, seed);
    case LLAISYS_DTYPE_BF16:
        return rand_sample_<llaisys::bf16_t>(sample_idx, sample_val, vals, 
            temperature, topK, topP, numel, batch_size, seed);
    case LLAISYS_DTYPE_F16:
        return rand_sample_<llaisys::fp16_t>(sample_idx, sample_val, vals, 
            temperature, topK, topP, numel, batch_size, seed);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
}
