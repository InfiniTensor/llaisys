/**
 * sample_cpu.cpp — CPU implementation of the random sampling operator
 *
 * This operator takes the raw logits output by the language model and samples
 * the next token using three optional filtering strategies:
 *
 *   Temperature scaling  — controls the "sharpness" of the distribution
 *   Top-K filtering      — restricts sampling to the K most likely tokens
 *   Top-P (nucleus)      — restricts sampling to the smallest set of tokens
 *                          whose cumulative probability exceeds P
 *
 * Algorithm overview (applied in order):
 *   1. Convert logits to float32 and divide by temperature
 *   2. Sort token indices by score (descending)
 *   3. Keep only the top-K indices (if top_k > 0)
 *   4. Compute softmax over the kept scores → probabilities
 *   5. Truncate to the nucleus (if top_p < 1.0) and re-normalize
 *   6. Draw one sample from the resulting discrete distribution
 *
 * References:
 *   Temperature: https://arxiv.org/abs/1904.09751
 *   Top-K:       https://arxiv.org/abs/1805.04833
 *   Top-P:       https://arxiv.org/abs/1904.09751
 */

#include "sample_cpu.hpp"

#include "../../../utils.hpp"

#include <cstdint>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

/**
 * Core sampling function, templated over the logit data type T.
 * Supports float32, float16, and bfloat16 via template specialization.
 *
 * @param sampled_token  Output: pointer to a single int64 that receives the
 *                       sampled token index.
 * @param logits         Input: raw unnormalized scores from the model's final
 *                       linear layer, one per vocabulary token.
 * @param vocab_size     Number of tokens in the vocabulary.
 * @param temperature    Divides all logits before softmax.
 *                       < 1.0 → sharper distribution (more confident)
 *                       > 1.0 → flatter distribution (more random)
 *                       = 1.0 → no change
 * @param top_k          Keep only the top-K tokens. 0 = disabled (keep all).
 * @param top_p          Nucleus threshold in [0, 1]. 1.0 = disabled (keep all).
 */
template <typename T>
void sample_(int64_t *sampled_token, const T *logits, size_t vocab_size,
             float temperature, int top_k, float top_p) {
    if (vocab_size == 0) {
        return;
    }

    // ── Step 1: Convert to float32 and apply temperature scaling ─────────────
    // Language models output logits in their native dtype (BF16, F16, or F32).
    // We convert everything to float32 for numerical stability.
    // Dividing by temperature before softmax is equivalent to raising the
    // probabilities to the power of (1/temperature) after softmax.
    std::vector<float> scores(vocab_size);
    for (size_t i = 0; i < vocab_size; i++) {
        float val;
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            val = llaisys::utils::cast<float>(logits[i]);
        } else {
            val = static_cast<float>(logits[i]);
        }
        scores[i] = val / temperature;
    }

    // ── Step 2: Sort token indices by score (descending) ─────────────────────
    // We sort an index array rather than the scores themselves so we can
    // map back to the original token IDs after filtering.
    std::vector<size_t> indices(vocab_size);
    std::iota(indices.begin(), indices.end(), 0);  // fill with 0, 1, 2, ...
    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        return scores[a] > scores[b];  // descending order
    });

    // ── Step 3: Top-K filtering ───────────────────────────────────────────────
    // Discard all but the K highest-scoring tokens.
    // Example: vocab_size=50000, top_k=50 → only 50 candidates remain.
    // top_k=0 means "keep all" (no filtering).
    // top_k=1 is equivalent to argmax (always picks the most likely token).
    size_t keep = vocab_size;
    if (top_k > 0 && static_cast<size_t>(top_k) < vocab_size) {
        keep = static_cast<size_t>(top_k);
    }
    indices.resize(keep);

    // ── Step 4: Numerically stable softmax over kept candidates ──────────────
    // Softmax converts raw scores to probabilities that sum to 1.
    // Formula: p_i = exp(s_i) / sum(exp(s_j))
    //
    // Numerical stability trick: subtract the maximum score before exp().
    // This prevents overflow (exp of large numbers → inf) without changing
    // the result, because the max cancels out in numerator and denominator:
    //   exp(s_i - max) / sum(exp(s_j - max))  ==  exp(s_i) / sum(exp(s_j))
    float max_score = scores[indices[0]];  // indices[0] is the highest score
    std::vector<float> probs(keep);
    float sum = 0.0f;
    for (size_t i = 0; i < keep; i++) {
        probs[i] = std::exp(scores[indices[i]] - max_score);
        sum += probs[i];
    }
    for (size_t i = 0; i < keep; i++) {
        probs[i] /= sum;  // normalize to sum to 1
    }

    // ── Step 5: Top-P (nucleus) filtering ────────────────────────────────────
    // Walk through tokens in probability order (already sorted from step 2).
    // Keep adding tokens until the cumulative probability reaches top_p,
    // then discard the rest. Re-normalize the kept probabilities.
    //
    // Example: top_p=0.9 with probs [0.5, 0.3, 0.15, 0.05]
    //   After token 0: cumsum = 0.5  (< 0.9, keep going)
    //   After token 1: cumsum = 0.8  (< 0.9, keep going)
    //   After token 2: cumsum = 0.95 (>= 0.9, stop here, cutoff = 3)
    //   Token 3 is discarded. Remaining probs re-normalized to sum to 1.
    if (top_p < 1.0f) {
        float cumsum = 0.0f;
        size_t cutoff = keep;
        for (size_t i = 0; i < keep; i++) {
            cumsum += probs[i];
            if (cumsum >= top_p) {
                cutoff = i + 1;
                break;
            }
        }
        if (cutoff < keep) {
            keep = cutoff;
            indices.resize(keep);
            probs.resize(keep);
            // Re-normalize so probabilities sum to 1 again
            float new_sum = 0.0f;
            for (float p : probs) new_sum += p;
            for (float &p : probs) p /= new_sum;
        }
    }

    // ── Step 6: Multinomial sampling ─────────────────────────────────────────
    // Draw one token index from the filtered probability distribution.
    // std::discrete_distribution handles the weighted random selection.
    //
    // thread_local: each thread gets its own RNG state, avoiding data races
    // in multi-threaded scenarios.
    //
    // Seed: XOR of std::random_device (hardware entropy) and the current
    // time. On Windows, std::random_device alone may return a fixed value,
    // so we mix in the clock to ensure different seeds across calls.
    thread_local std::mt19937 rng(
        std::random_device{}() ^
        static_cast<uint32_t>(std::chrono::steady_clock::now().time_since_epoch().count())
    );
    std::discrete_distribution<size_t> dist(probs.begin(), probs.end());
    *sampled_token = static_cast<int64_t>(indices[dist(rng)]);
}

/**
 * Public entry point for the sample operator.
 * Dispatches to the templated sample_() based on the logit data type.
 * Raw byte pointers are used to match the generic tensor data interface.
 */
namespace llaisys::ops::cpu {
void sample(std::byte *sampled_token, const std::byte *logits, llaisysDataType_t logits_dtype,
            size_t vocab_size, float temperature, int top_k, float top_p) {
    int64_t *token_ptr = reinterpret_cast<int64_t *>(sampled_token);

    switch (logits_dtype) {
    case LLAISYS_DTYPE_F32:
        sample_(token_ptr, reinterpret_cast<const float *>(logits), vocab_size, temperature, top_k, top_p);
        return;
    case LLAISYS_DTYPE_BF16:
        sample_(token_ptr, reinterpret_cast<const llaisys::bf16_t *>(logits), vocab_size, temperature, top_k, top_p);
        return;
    case LLAISYS_DTYPE_F16:
        sample_(token_ptr, reinterpret_cast<const llaisys::fp16_t *>(logits), vocab_size, temperature, top_k, top_p);
        return;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(logits_dtype);
    }
}
} // namespace llaisys::ops::cpu
