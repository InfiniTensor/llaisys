#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {

/**
 * @brief Sample a token from logits with optional temperature, top-k, and top-p filtering.
 *
 * @param out   Output tensor, shape {1}, dtype int64. Receives the sampled token index.
 * @param logits Input logits tensor, shape {vocab_size}, any float dtype.
 * @param top_k  Keep only the top-k highest-logit tokens before sampling (0 = disabled).
 * @param top_p  Nucleus sampling: keep the smallest set of tokens whose cumulative probability
 *               exceeds top_p (1.0 = disabled).
 * @param temperature Divide logits by this value before softmax (1.0 = no change).
 */
void sample(tensor_t out, tensor_t logits, int top_k, float top_p, float temperature);

/**
 * @brief Set the random seed for the sampling RNG (per-thread).
 *        Call before sampling for reproducible results.
 */
void sample_set_seed(uint64_t seed);

} // namespace llaisys::ops
