#ifndef LLAISYS_MODELS_QWEN2_H
#define LLAISYS_MODELS_QWEN2_H

#include "../tensor.h"

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

    __export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice);

    __export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model);

    __export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model);

    // Set a named weight tensor into the model. Returns 0 on success.
    __export int llaisysQwen2ModelSetWeight(struct LlaisysQwen2Model * model, const char * name, llaisysTensor_t tensor);

    // Optional finalize call after all weights are set.
    __export int llaisysQwen2ModelFinalize(struct LlaisysQwen2Model * model);

    // Check whether a named weight has been set. Returns 1 if present, 0 otherwise.
    __export uint8_t llaisysQwen2ModelHasWeight(struct LlaisysQwen2Model * model, const char * name);

    __export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken);
    __export int64_t llaisysQwen2ModelInferSample(
        struct LlaisysQwen2Model * model,
        int64_t * token_ids,
        size_t ntoken,
        float temperature,
        size_t top_k,
        float top_p,
        uint64_t seed);

    // KV cache APIs
    __export void *llaisysQwen2KVCreat(struct LlaisysQwen2Model * model, size_t max_tokens);
    __export void llaisysQwen2KVDestroy(void *kv);
    __export int llaisysQwen2KVAppend(void *kv, llaisysTensor_t k, llaisysTensor_t v);
    __export size_t llaisysQwen2KVLen(void *kv);
}
#endif // LLAISYS_MODELS_QWEN2_H
