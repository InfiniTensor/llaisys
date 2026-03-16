#ifdef __AVX2__
#include <immintrin.h>
#endif

#include "sample_cpu.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

static thread_local std::mt19937 rng{std::random_device{}()};

template <typename T>
void sample_(int64_t *out_idx, const T *logits, size_t numel,
             float temperature, int top_k, float top_p) {
    std::vector<float> probs(numel);
    for (size_t i = 0; i < numel; i++) {
        probs[i] = llaisys::utils::cast<float>(logits[i]);
    }

    if (temperature <= 0.0f) temperature = 1.0f;
    if (temperature != 1.0f) {
        for (size_t i = 0; i < numel; i++) {
            probs[i] /= temperature;
        }
    }

    // Build index array sorted by descending logit value
    std::vector<int> indices(numel);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&](int a, int b) { return probs[a] > probs[b]; });

    // Top-K: keep at most top_k candidates
    size_t keep = numel;
    if (top_k > 0 && static_cast<size_t>(top_k) < numel) {
        keep = static_cast<size_t>(top_k);
    }

    // Softmax over the kept candidates
    float max_val = probs[indices[0]];
    std::vector<float> softmax_vals(keep);
    float sum_exp = 0.0f;
    for (size_t i = 0; i < keep; i++) {
        softmax_vals[i] = std::exp(probs[indices[i]] - max_val);
        sum_exp += softmax_vals[i];
    }
    for (size_t i = 0; i < keep; i++) {
        softmax_vals[i] /= sum_exp;
    }

    // Top-P (nucleus): find cutoff where cumulative prob >= top_p
    if (top_p > 0.0f && top_p < 1.0f) {
        float cumsum = 0.0f;
        size_t cutoff = keep;
        for (size_t i = 0; i < keep; i++) {
            cumsum += softmax_vals[i];
            if (cumsum >= top_p) {
                cutoff = i + 1;
                break;
            }
        }
        keep = cutoff;
        // Re-normalize
        float new_sum = 0.0f;
        for (size_t i = 0; i < keep; i++) new_sum += softmax_vals[i];
        for (size_t i = 0; i < keep; i++) softmax_vals[i] /= new_sum;
    }

    // Sample from the distribution
    std::discrete_distribution<int> dist(softmax_vals.begin(), softmax_vals.begin() + keep);
    int sampled = dist(rng);
    *out_idx = static_cast<int64_t>(indices[sampled]);
}

namespace llaisys::ops::cpu {
void sample(std::byte *out_idx, const std::byte *logits, llaisysDataType_t type, size_t numel,
            float temperature, int top_k, float top_p) {
    auto *idx_ptr = reinterpret_cast<int64_t *>(out_idx);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return sample_(idx_ptr, reinterpret_cast<const float *>(logits), numel, temperature, top_k, top_p);
    case LLAISYS_DTYPE_BF16:
        return sample_(idx_ptr, reinterpret_cast<const llaisys::bf16_t *>(logits), numel, temperature, top_k, top_p);
    case LLAISYS_DTYPE_F16:
        return sample_(idx_ptr, reinterpret_cast<const llaisys::fp16_t *>(logits), numel, temperature, top_k, top_p);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
