#ifndef LLAISYS_MODELS_QWEN2_H
#define LLAISYS_MODELS_QWEN2_H

#include "../tensor.h"
#include "llaisys.h"
#include <cstddef>

__LLAISYS__C {
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

    struct Qwen2TensorPool {
        // 每层的中间张量 (复用于所有推理步骤)
        llaisysTensor_t *attn_layernorm;  // [nlayer] 每个: (max_seqlen, hs)
        llaisysTensor_t *q_proj;          // [nlayer] 每个: (max_seqlen, hs)
        llaisysTensor_t *q_rope;          // [nlayer] 每个: (max_seqlen, nh, dh)
        llaisysTensor_t *k_proj;          // [nlayer] 每个: (max_seqlen, dkvh)
        // llaisysTensor_t *k_proj_viewed;   // [nlayer] 每个: (max_seqlen, nkvh, dh) - view only
        llaisysTensor_t *attn_val;        // [nlayer] 每个: (max_seqlen, nh, dh)
        // llaisysTensor_t *attn_val_viewed; // [nlayer] 每个: (max_seqlen, hs) - view only
        llaisysTensor_t *o_proj;          // [nlayer] 每个: (max_seqlen, hs)
        llaisysTensor_t *mlp_layer;       // [nlayer] 每个: (max_seqlen, hs)
        llaisysTensor_t *mlp_layernorm;   // [nlayer] 每个: (max_seqlen, hs)
        llaisysTensor_t *mlp_gate;        // [nlayer] 每个: (max_seqlen, di)
        llaisysTensor_t *mlp_up;          // [nlayer] 每个: (max_seqlen, di)
        llaisysTensor_t *mlp_swiglu;      // [nlayer] 每个: (max_seqlen, di)
        llaisysTensor_t *mlp_down;        // [nlayer] 每个: (max_seqlen, hs)

        // 全局中间张量 (跨层复用)
        llaisysTensor_t pos_ids;          // (max_seqlen)
        llaisysTensor_t input_token;      // (max_seqlen)
        llaisysTensor_t input_embed;      // (max_seqlen, hs)
        llaisysTensor_t output_layernorm; // (max_seqlen, hs)
        llaisysTensor_t output_embed;     // (max_seqlen, voc)
        llaisysTensor_t max_idx;          // (1)
        llaisysTensor_t max_vals;         // (1)

        size_t max_seqlen; // 池支持的最大序列长度
        bool initialized;  // 是否已初始化
    };

    struct LlaisysQwen2KVCache {
        llaisysTensor_t *kcache;
        llaisysTensor_t *vcache;
    };

    struct LlaisysQwen2Model {
        struct LlaisysQwen2Meta meta;
        struct LlaisysQwen2Weights weights;
        struct LlaisysQwen2KVCache kvcache;
        llaisysDeviceType_t device;
        int ndevice;
        int *device_ids;
        struct Qwen2TensorPool *tensor_pool = NULL;
    };

    __export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, const LlaisysQwen2Weights *weights, llaisysDeviceType_t device, int ndevice, int *device_ids);

    __export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model);

    __export struct LlaisysQwen2KVCache *llaisysQwen2KVCacheCreate(struct LlaisysQwen2Model * meta, size_t max_len);

    __export void llaisysQwen2KVCacheDestroy(struct LlaisysQwen2KVCache * kvcache, size_t nlayer);

    __export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t *token_ids, size_t ntoken, struct LlaisysQwen2KVCache *kvcache, size_t past_len);
}
#endif // LLAISYS_MODELS_QWEN2_H
