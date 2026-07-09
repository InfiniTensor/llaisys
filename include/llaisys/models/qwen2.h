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
    struct LlaisysQwen2Session;

    __export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice);

    __export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model);

    __export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model);

    // ── 向后兼容：操作模型内置 default_session
    __export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t *token_ids, size_t ntoken);

    __export int64_t llaisysQwen2ModelInferSample(struct LlaisysQwen2Model * model, int64_t *token_ids, size_t ntoken, float temperature, int top_k, float top_p);

    __export void llaisysQwen2ModelSetCachePos(struct LlaisysQwen2Model * model, size_t pos);

    __export size_t llaisysQwen2ModelGetCachePos(struct LlaisysQwen2Model * model);

    __export void llaisysQwen2ModelResetCache(struct LlaisysQwen2Model * model);

    // ── 多用户 Session API：每用户独占 KV-Cache
    __export struct LlaisysQwen2Session *llaisysQwen2SessionCreate(struct LlaisysQwen2Model * model);

    __export void llaisysQwen2SessionDestroy(struct LlaisysQwen2Session * session);

    __export int64_t llaisysQwen2SessionInfer(struct LlaisysQwen2Model * model, struct LlaisysQwen2Session * session, int64_t *token_ids, size_t ntoken);

    __export int64_t llaisysQwen2SessionInferSample(struct LlaisysQwen2Model * model, struct LlaisysQwen2Session * session, int64_t *token_ids, size_t ntoken, float temperature, int top_k, float top_p);

    __export void llaisysQwen2SessionSetCachePos(struct LlaisysQwen2Session * session, size_t pos);

    __export size_t llaisysQwen2SessionGetCachePos(struct LlaisysQwen2Session * session);

    __export void llaisysQwen2SessionResetCache(struct LlaisysQwen2Session * session);
}
#endif // LLAISYS_MODELS_QWEN2_H
