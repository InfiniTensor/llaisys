#ifndef LLAISYS_OPS_H
#define LLAISYS_OPS_H

#include "tensor.h"

__C {
    __export void llaisysAdd(llaisysTensor_t c, llaisysTensor_t a, llaisysTensor_t b);
    __export void llaisysArgmax(llaisysTensor_t max_idx, llaisysTensor_t max_val, llaisysTensor_t vals);
    __export void llaisysEmbedding(llaisysTensor_t out, llaisysTensor_t index, llaisysTensor_t weight);
    __export void llaisysLinear(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t weight, llaisysTensor_t bias);
    __export void llaisysRearrange(llaisysTensor_t out, llaisysTensor_t in);
    __export void llaisysRmsNorm(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t weight, float eps);
    __export void llaisysROPE(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t pos_ids, float theta);
    __export void llaisysSelfAttention(llaisysTensor_t attn_val, llaisysTensor_t q, llaisysTensor_t k, llaisysTensor_t v, float scale);
    __export void llaisysSwiGLU(llaisysTensor_t out, llaisysTensor_t gate, llaisysTensor_t up);

    /**
     * @brief Sample a token index from logits with temperature / top-k / top-p.
     * @param out       Shape {1}, dtype int64. Receives the sampled token index.
     * @param logits    Shape {vocab_size}, any float dtype. Must be contiguous.
     * @param top_k     Keep only top-k tokens (0 = disabled).
     * @param top_p     Nucleus threshold in (0, 1] (1.0 = disabled).
     * @param temperature Positive temperature scalar.
     */
    __export void llaisysSample(llaisysTensor_t out, llaisysTensor_t logits,
                                int top_k, float top_p, float temperature);

    /** Set the per-thread RNG seed used by llaisysSample. */
    __export void llaisysSampleSetSeed(uint64_t seed);
}

#endif
