#ifndef LLAISYS_MODELS_QWEN2_H
#define LLAISYS_MODELS_QWEN2_H

#include "../tensor.h"

__C {
    // Qwen2 model meta info
    struct LlaisysQwen2Meta {
        // Data type of the model weights. Only supports int8 and float16 for now.
        llaisysDataType_t dtype;
        // Model hyperparameters
        // nlayer: number of layers
        // hs: hidden size
        // nh: number of attention heads（Q头）
        // nkvh: number of key/value heads
        // dh: head dimension
        // di: intermediate dimension
        // maxseq: maximum sequence length
        // voc: vocabulary size
        size_t nlayer, hs, nh, nkvh, dh, di, maxseq, voc;
        // Sampling parameters
        // epsilon: sampling parameter epsilon
        // theta: sampling parameter theta
        // end_token: end token ID
        float epsilon, theta;
        int64_t end_token;
    };

    // Forward declaration of the model implementation, which is hidden from the API users.
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

    enum LlaisysQwen2WeightKind {
        LLAISYS_QWEN2_WEIGHT_IN_EMBED = 0,
        LLAISYS_QWEN2_WEIGHT_OUT_EMBED = 1,
        LLAISYS_QWEN2_WEIGHT_OUT_NORM = 2,
        LLAISYS_QWEN2_WEIGHT_ATTN_NORM = 3,
        LLAISYS_QWEN2_WEIGHT_ATTN_Q_W = 4,
        LLAISYS_QWEN2_WEIGHT_ATTN_Q_B = 5,
        LLAISYS_QWEN2_WEIGHT_ATTN_K_W = 6,
        LLAISYS_QWEN2_WEIGHT_ATTN_K_B = 7,
        LLAISYS_QWEN2_WEIGHT_ATTN_V_W = 8,
        LLAISYS_QWEN2_WEIGHT_ATTN_V_B = 9,
        LLAISYS_QWEN2_WEIGHT_ATTN_O_W = 10,
        LLAISYS_QWEN2_WEIGHT_MLP_NORM = 11,
        LLAISYS_QWEN2_WEIGHT_MLP_GATE_W = 12,
        LLAISYS_QWEN2_WEIGHT_MLP_UP_W = 13,
        LLAISYS_QWEN2_WEIGHT_MLP_DOWN_W = 14
    };

    __export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice);

    __export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model);

    __export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model);

    __export llaisysTensor_t llaisysQwen2ModelGetWeight(struct LlaisysQwen2Model * model, int kind, size_t layer);

    __export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t *token_ids, size_t ntoken);
}
#endif // LLAISYS_MODELS_QWEN2_H
