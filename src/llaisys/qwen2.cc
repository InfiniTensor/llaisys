#include "llaisys/models/qwen2.h"

#include "llaisys_tensor.hpp"
#include "../models/qwen2/qwen2_model.hpp"
#include "../utils.hpp"

struct LlaisysQwen2Model {
    llaisys::models::qwen2::Qwen2Model *impl = nullptr;
    LlaisysQwen2Meta meta{};
    LlaisysQwen2Weights weights{};
    llaisysDeviceType_t device = LLAISYS_DEVICE_CPU;
    int device_id = 0;
};

namespace {
void init_layer_arrays(LlaisysQwen2Weights &weights, size_t nlayer) {
    weights.in_embed = nullptr;
    weights.out_embed = nullptr;
    weights.out_norm_w = nullptr;

    weights.attn_norm_w = new llaisysTensor_t[nlayer]();
    weights.attn_q_w = new llaisysTensor_t[nlayer]();
    weights.attn_q_b = new llaisysTensor_t[nlayer]();
    weights.attn_k_w = new llaisysTensor_t[nlayer]();
    weights.attn_k_b = new llaisysTensor_t[nlayer]();
    weights.attn_v_w = new llaisysTensor_t[nlayer]();
    weights.attn_v_b = new llaisysTensor_t[nlayer]();
    weights.attn_o_w = new llaisysTensor_t[nlayer]();

    weights.mlp_norm_w = new llaisysTensor_t[nlayer]();
    weights.mlp_gate_w = new llaisysTensor_t[nlayer]();
    weights.mlp_up_w = new llaisysTensor_t[nlayer]();
    weights.mlp_down_w = new llaisysTensor_t[nlayer]();
}

void free_layer_arrays(LlaisysQwen2Weights &weights) {
    delete[] weights.attn_norm_w;
    delete[] weights.attn_q_w;
    delete[] weights.attn_q_b;
    delete[] weights.attn_k_w;
    delete[] weights.attn_k_b;
    delete[] weights.attn_v_w;
    delete[] weights.attn_v_b;
    delete[] weights.attn_o_w;

    delete[] weights.mlp_norm_w;
    delete[] weights.mlp_gate_w;
    delete[] weights.mlp_up_w;
    delete[] weights.mlp_down_w;

    weights.attn_norm_w = nullptr;
    weights.attn_q_w = nullptr;
    weights.attn_q_b = nullptr;
    weights.attn_k_w = nullptr;
    weights.attn_k_b = nullptr;
    weights.attn_v_w = nullptr;
    weights.attn_v_b = nullptr;
    weights.attn_o_w = nullptr;

    weights.mlp_norm_w = nullptr;
    weights.mlp_gate_w = nullptr;
    weights.mlp_up_w = nullptr;
    weights.mlp_down_w = nullptr;
}
} // namespace

__C {
    struct LlaisysQwen2Model *llaisysQwen2ModelCreate(
        const LlaisysQwen2Meta *meta,
        llaisysDeviceType_t device,
        int *device_ids,
        int ndevice) {
        CHECK_ARGUMENT(meta != nullptr, "meta is null");

        auto *model = new LlaisysQwen2Model();
        model->meta = *meta;
        model->device = device;
        model->device_id = (device_ids && ndevice > 0) ? device_ids[0] : 0;

        init_layer_arrays(model->weights, model->meta.nlayer);
        model->impl = new llaisys::models::qwen2::Qwen2Model(model->meta, device, model->device_id);

        return model;
    }

    void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
        if (!model) {
            return;
        }
        delete model->impl;
        model->impl = nullptr;
        free_layer_arrays(model->weights);
        delete model;
    }

    struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
        CHECK_ARGUMENT(model != nullptr, "model is null");
        return &model->weights;
    }

    int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
        CHECK_ARGUMENT(model != nullptr, "model is null");
        model->impl->bind_weights(model->weights);
        return model->impl->infer(token_ids, ntoken);
    }
}
