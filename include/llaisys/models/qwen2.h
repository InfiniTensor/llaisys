#ifndef LLAISYS_MODELS_QWEN2_H
#define LLAISYS_MODELS_QWEN2_H

#include "../tensor.h"
#include <vector>

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
        llaisysTensor_t out_norm_w;
        // 改为 vector
        std::vector<llaisysTensor_t> attn_norm_w; 
        std::vector<llaisysTensor_t> attn_q_w;
        std::vector<llaisysTensor_t> attn_q_b;
        std::vector<llaisysTensor_t> attn_k_w;
        std::vector<llaisysTensor_t> attn_k_b;
        std::vector<llaisysTensor_t> attn_v_w;
        std::vector<llaisysTensor_t> attn_v_b;
        std::vector<llaisysTensor_t> attn_o_w;
        std::vector<llaisysTensor_t> mlp_norm_w;
        std::vector<llaisysTensor_t> mlp_gate_w;
        std::vector<llaisysTensor_t> mlp_up_w;
        std::vector<llaisysTensor_t> mlp_down_w;
    };

    struct LlaisysQwen2Model;

    __export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice);

    __export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model);

    __export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model);

    __export void llaisysQwen2LoadWeight(
        struct LlaisysQwen2Model * model,
        const char * name,
        const void * data,
        size_t * shape,
        size_t ndim,
        llaisysDataType_t dtype);

    __export int64_t llaisysQwen2ModelInfer(
        struct LlaisysQwen2Model * model,
        int64_t * token_ids,
        size_t ntoken,
        size_t start_pos);
}
#endif // LLAISYS_MODELS_QWEN2_H
