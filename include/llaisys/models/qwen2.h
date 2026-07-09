#ifndef LLAISYS_MODELS_QWEN2_H
#define LLAISYS_MODELS_QWEN2_H

#include "../tensor.h"

__C {
    typedef struct LlaisysQwen2Meta_ {
        llaisysDataType_t dtype;
        size_t nlayer, hs, nh, nkvh, dh, di, maxseq, voc;
        float epsilon, theta;
        int64_t end_token;
    }LlaisysQwen2Meta;

    typedef struct LlaisysQwen2Weights_ {
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
    }LlaisysQwen2Weights;

    typedef struct LlaisysQwen2Model_ {
        LlaisysQwen2Meta* meta;
        LlaisysQwen2Weights* weights = nullptr;
        void *impl = nullptr; // Opaque pointer to the actual model implementation (e.g., a C++ class instance).
    }LlaisysQwen2Model;



    __export LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice);

    __export void llaisysQwen2ModelDestroy(LlaisysQwen2Model * model);
    
    __export void llaisysQwen2modelLoadWeight(LlaisysQwen2Model * model, const void *weight_data, const char *weight_name);

    __export LlaisysQwen2Weights *llaisysQwen2ModelWeights(LlaisysQwen2Model * model);

    __export int64_t llaisysQwen2ModelInfer(LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken);
}
#endif // LLAISYS_MODELS_QWEN2_H
