#ifndef LLAISYS_MODELS_QWEN2_H
#define LLAISYS_MODELS_QWEN2_H

#include "../tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

// 模型超参数元数据
typedef struct {
    int dtype;         // 0=F32, 1=F16...
    size_t nlayer;     // 层数
    size_t hs;         // Hidden Size
    size_t nh;         // Num Attention Heads
    size_t nkvh;       // Num KV Heads
    size_t dh;         // Head Dim (hs / nh)
    size_t di;         // Intermediate Size (FFN)
    size_t maxseq;     // Max Position Embeddings
    size_t voc;        // Vocab Size
    float epsilon;     // RMS Norm Epsilon
    float theta;       // RoPE Theta
    int64_t end_token; // EOS Token ID
} LlaisysQwen2Meta;

// 权重指针容器 (C++端分配数组，Python端填充数据)
typedef struct {
    llaisysTensor_t in_embed;
    llaisysTensor_t out_embed;
    llaisysTensor_t out_norm_w;

    // 以下是指针数组 (Array of Tensors)，长度为 nlayer
    llaisysTensor_t *attn_norm_w;
    llaisysTensor_t *attn_q_w;
    llaisysTensor_t *attn_q_b;
    llaisysTensor_t *attn_k_w;
    llaisysTensor_t *attn_k_b;
    llaisysTensor_t *attn_v_w;
    llaisysTensor_t *attn_v_b;
    llaisysTensor_t *attn_o_w; // Qwen 通常无 o_bias

    llaisysTensor_t *mlp_norm_w;
    llaisysTensor_t *mlp_gate_w;
    llaisysTensor_t *mlp_up_w;
    llaisysTensor_t *mlp_down_w;
} LlaisysQwen2Weights;

// 不透明模型句柄
struct LlaisysQwen2Model;

// API 导出
__export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice);

__export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model);

__export LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model);

__export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken, float temperature, float top_p, size_t top_k);

#ifdef __cplusplus
}
#endif

#endif // LLAISYS_MODELS_QWEN2_H
