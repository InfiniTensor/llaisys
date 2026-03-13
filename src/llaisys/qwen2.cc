#include "llaisys/models/qwen2.h"
#include "../models/qwen2/qwen2.hpp"
#include "llaisys_tensor.hpp"

__C {

struct LlaisysQwen2Model {
    llaisys::models::Qwen2Model* model;
    LlaisysQwen2Weights weights;
};

__export struct LlaisysQwen2Model* llaisysQwen2ModelCreate(
    const LlaisysQwen2Meta* meta,
    llaisysDeviceType_t device,
    int* device_ids,
    int ndevice) {

    auto* wrapper = new LlaisysQwen2Model();
    int device_id = (device_ids && ndevice > 0) ? device_ids[0] : 0;
    wrapper->model = new llaisys::models::Qwen2Model(meta, device, device_id);

    // Set up weights pointers
    auto& m = *wrapper->model;
    auto& w = wrapper->weights;
    size_t nlayer = meta->nlayer;

    w.in_embed = new LlaisysTensor{m.in_embed};
    w.out_embed = new LlaisysTensor{m.out_embed};
    w.out_norm_w = new LlaisysTensor{m.out_norm_w};

    w.attn_norm_w = new llaisysTensor_t[nlayer];
    w.attn_q_w = new llaisysTensor_t[nlayer];
    w.attn_q_b = new llaisysTensor_t[nlayer];
    w.attn_k_w = new llaisysTensor_t[nlayer];
    w.attn_k_b = new llaisysTensor_t[nlayer];
    w.attn_v_w = new llaisysTensor_t[nlayer];
    w.attn_v_b = new llaisysTensor_t[nlayer];
    w.attn_o_w = new llaisysTensor_t[nlayer];
    w.mlp_norm_w = new llaisysTensor_t[nlayer];
    w.mlp_gate_w = new llaisysTensor_t[nlayer];
    w.mlp_up_w = new llaisysTensor_t[nlayer];
    w.mlp_down_w = new llaisysTensor_t[nlayer];

    for (size_t i = 0; i < nlayer; ++i) {
        w.attn_norm_w[i] = new LlaisysTensor{m.attn_norm_w[i]};
        w.attn_q_w[i] = new LlaisysTensor{m.attn_q_w[i]};
        w.attn_q_b[i] = new LlaisysTensor{m.attn_q_b[i]};
        w.attn_k_w[i] = new LlaisysTensor{m.attn_k_w[i]};
        w.attn_k_b[i] = new LlaisysTensor{m.attn_k_b[i]};
        w.attn_v_w[i] = new LlaisysTensor{m.attn_v_w[i]};
        w.attn_v_b[i] = new LlaisysTensor{m.attn_v_b[i]};
        w.attn_o_w[i] = new LlaisysTensor{m.attn_o_w[i]};
        w.mlp_norm_w[i] = new LlaisysTensor{m.mlp_norm_w[i]};
        w.mlp_gate_w[i] = new LlaisysTensor{m.mlp_gate_w[i]};
        w.mlp_up_w[i] = new LlaisysTensor{m.mlp_up_w[i]};
        w.mlp_down_w[i] = new LlaisysTensor{m.mlp_down_w[i]};
    }

    return wrapper;
}

__export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model* model) {
    if (!model) return;

    size_t nlayer = model->model->meta.nlayer;
    auto& w = model->weights;

    delete w.in_embed;
    delete w.out_embed;
    delete w.out_norm_w;

    for (size_t i = 0; i < nlayer; ++i) {
        delete w.attn_norm_w[i];
        delete w.attn_q_w[i];
        delete w.attn_q_b[i];
        delete w.attn_k_w[i];
        delete w.attn_k_b[i];
        delete w.attn_v_w[i];
        delete w.attn_v_b[i];
        delete w.attn_o_w[i];
        delete w.mlp_norm_w[i];
        delete w.mlp_gate_w[i];
        delete w.mlp_up_w[i];
        delete w.mlp_down_w[i];
    }

    delete[] w.attn_norm_w;
    delete[] w.attn_q_w;
    delete[] w.attn_q_b;
    delete[] w.attn_k_w;
    delete[] w.attn_k_b;
    delete[] w.attn_v_w;
    delete[] w.attn_v_b;
    delete[] w.attn_o_w;
    delete[] w.mlp_norm_w;
    delete[] w.mlp_gate_w;
    delete[] w.mlp_up_w;
    delete[] w.mlp_down_w;

    delete model->model;
    delete model;
}

__export struct LlaisysQwen2Weights* llaisysQwen2ModelWeights(struct LlaisysQwen2Model* model) {
    return &model->weights;
}

__export int64_t llaisysQwen2ModelInfer(
    struct LlaisysQwen2Model* model,
    int64_t* token_ids,
    size_t ntoken) {
    return model->model->infer(token_ids, ntoken);
}

__export void llaisysQwen2ModelResetCache(struct LlaisysQwen2Model* model) {
    model->model->cache_len = 0;
}

}
