#ifndef LLAISYS_MODELS_QWEN2_H
#define LLAISYS_MODELS_QWEN2_H

#include "../tensor.h"
#include <cstdint>
#include <string>

__C {
    struct LlaisysQwen2Meta {
        llaisysDataType_t dtype;
        size_t nlayer, hs, nh, nkvh, dh, di, maxseq, voc;
        float epsilon, theta;
        int64_t end_token;
    };

    struct LlaisysQwen2Weights {
        llaisysTensor_t in_embed;
        llaisysTensor_t out_embed;
        llaisysTensor_t out_norm_w;   // a.k.a. model.norm.weight
        llaisysTensor_t *attn_norm_w; // a.k.a. input_layernorm.weight
        llaisysTensor_t *attn_q_w;
        llaisysTensor_t *attn_q_b;
        llaisysTensor_t *attn_k_w;
        llaisysTensor_t *attn_k_b;
        llaisysTensor_t *attn_v_w;
        llaisysTensor_t *attn_v_b;
        llaisysTensor_t *attn_o_w;
        llaisysTensor_t *mlp_norm_w; // a.k.a. post_attention_layernorm.weight
        llaisysTensor_t *mlp_gate_w;
        llaisysTensor_t *mlp_up_w;
        llaisysTensor_t *mlp_down_w;
    };

    struct LlaisysQwen2Model;

    __export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(
        const LlaisysQwen2Meta *meta, llaisysDeviceType_t device,
        int *device_ids, int ndevice);

    __export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model);

    __export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(
        struct LlaisysQwen2Model * model);

        /**
         * @brief Inference function for Qwen2 Model. This function combines both
         * prefill and decode through `prefill` flag.
         * @note This function will reset KV Caches if `prefill` is true.
         * 
         * @param token_ids input token ids
         * @param pos_ids input position ids, used for RoPE
         */
    __export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model,
                                            int64_t *token_ids, int64_t *pos_ids, size_t ntoken, bool prefill);

    /**
     * @brief Inference with configurable decoding strategy.
     *
     * This behaves the same as llaisysQwen2ModelInfer for prefill/decode flow,
     * but selects the next token using temperature/top-k/top-p sampling.
     *
     * @param top_k        Keep only top-k tokens before sampling (0 = disabled).
     * @param top_p        Nucleus threshold in (0, 1] (1.0 = disabled).
     * @param temperature  Positive temperature scalar.
     */
    __export int64_t llaisysQwen2ModelInferSample(struct LlaisysQwen2Model * model,
                                                  int64_t *token_ids, int64_t *pos_ids,
                                                  size_t ntoken, bool prefill,
                                                  int top_k, float top_p, float temperature);

    __export void llaisysQwen2SetWeights(struct LlaisysQwen2Model * model,
                                         int name, int layer_id,
                                         llaisysTensor_t tensor);
}
#endif // LLAISYS_MODELS_QWEN2_H
