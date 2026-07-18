#ifndef LLAISYS_QWEN2_H
#define LLAISYS_QWEN2_H

#include "../llaisys.h"
#include "tensor.h"

__C {
    typedef struct LlaisysQwen2Model *llaisysQwen2Model_t;

    struct LlaisysQwen2Meta {
        size_t nlayer;           // num_hidden_layers
        size_t hs;               // hidden_size
        size_t nh;               // num_attention_heads
        size_t nkvh;             // num_key_value_heads
        size_t dh;               // head_dim = hs / nh
        size_t di;               // intermediate_size
        size_t maxseq;           // max_position_embeddings
        size_t voc;              // vocab_size
        float epsilon;           // rms_norm_eps
        float theta;             // rope_theta
        int64_t end_token;       // eos_token_id
    };

    struct LlaisysQwen2Weights {
        llaisysTensor_t in_embed;       // [voc, hs]
        llaisysTensor_t out_embed;      // [voc, hs]
        llaisysTensor_t out_norm_w;     // [hs]

        // Per-layer weights (arrays of size nlayer)
        llaisysTensor_t* attn_norm_w;   // [nlayer][hs]
        llaisysTensor_t* attn_q_w;      // [nlayer][nh*dh, hs]
        llaisysTensor_t* attn_q_b;      // [nlayer][nh*dh]
        llaisysTensor_t* attn_k_w;      // [nlayer][nkvh*dh, hs]
        llaisysTensor_t* attn_k_b;      // [nlayer][nkvh*dh]
        llaisysTensor_t* attn_v_w;      // [nlayer][nkvh*dh, hs]
        llaisysTensor_t* attn_v_b;      // [nlayer][nkvh*dh]
        llaisysTensor_t* attn_o_w;      // [nlayer][hs, nh*dh]

        llaisysTensor_t* mlp_norm_w;    // [nlayer][hs]
        llaisysTensor_t* mlp_gate_w;    // [nlayer][di, hs]
        llaisysTensor_t* mlp_up_w;      // [nlayer][di, hs]
        llaisysTensor_t* mlp_down_w;    // [nlayer][hs, di]
    };

    __export llaisysQwen2Model_t llaisysQwen2ModelCreate(
        const struct LlaisysQwen2Meta* meta,
        llaisysDeviceType_t device_type,
        int device_id);

    __export void llaisysQwen2ModelDestroy(llaisysQwen2Model_t model);

    __export struct LlaisysQwen2Weights* llaisysQwen2ModelWeights(llaisysQwen2Model_t model);

    __export int64_t llaisysQwen2ModelInfer(
        llaisysQwen2Model_t model,
        int64_t* token_ids,
        size_t ntoken);

    __export void llaisysQwen2ModelResetCache(llaisysQwen2Model_t model);
}

#endif // LLAISYS_QWEN2_H
