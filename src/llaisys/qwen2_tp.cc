#include "llaisys/models/qwen2_tp.h"
#include "../models/qwen2/qwen2_tp.hpp"
#include "llaisys_tensor.hpp"

#include <vector>

__C {

struct LlaisysQwen2ModelTP {
    llaisys::models::Qwen2ModelTP* model;
    std::vector<LlaisysQwen2Weights*> weights_per_rank;
};

__export struct LlaisysQwen2ModelTP* llaisysQwen2ModelTPCreate(
    const struct LlaisysQwen2Meta* meta,
    const int* device_ids,
    int world_size) {

    // Convert device_ids to vector
    std::vector<int> device_vec(device_ids, device_ids + world_size);

    auto* wrapper = new LlaisysQwen2ModelTP();
    wrapper->model = new llaisys::models::Qwen2ModelTP(meta, device_vec);

    // Get weights for each rank
    wrapper->weights_per_rank.resize(world_size);
    for (int i = 0; i < world_size; ++i) {
        wrapper->weights_per_rank[i] = wrapper->model->getWeights(i);
    }

    return wrapper;
}

__export void llaisysQwen2ModelTPDestroy(struct LlaisysQwen2ModelTP* model) {
    if (!model) return;

    // Clean up weights for each rank
    for (auto* weights : model->weights_per_rank) {
        if (!weights) continue;

        size_t nlayer = model->model->meta.nlayer;

        delete weights->in_embed;
        delete weights->out_embed;
        delete weights->out_norm_w;

        for (size_t i = 0; i < nlayer; ++i) {
            delete weights->attn_norm_w[i];
            delete weights->attn_q_w[i];
            delete weights->attn_q_b[i];
            delete weights->attn_k_w[i];
            delete weights->attn_k_b[i];
            delete weights->attn_v_w[i];
            delete weights->attn_v_b[i];
            delete weights->attn_o_w[i];
            delete weights->mlp_norm_w[i];
            delete weights->mlp_gate_w[i];
            delete weights->mlp_up_w[i];
            delete weights->mlp_down_w[i];
        }

        delete[] weights->attn_norm_w;
        delete[] weights->attn_q_w;
        delete[] weights->attn_q_b;
        delete[] weights->attn_k_w;
        delete[] weights->attn_k_b;
        delete[] weights->attn_v_w;
        delete[] weights->attn_v_b;
        delete[] weights->attn_o_w;
        delete[] weights->mlp_norm_w;
        delete[] weights->mlp_gate_w;
        delete[] weights->mlp_up_w;
        delete[] weights->mlp_down_w;

        delete weights;
    }

    delete model->model;
    delete model;
}

__export struct LlaisysQwen2Weights* llaisysQwen2ModelTPWeights(
    struct LlaisysQwen2ModelTP* model,
    int rank) {
    if (!model || rank < 0 || rank >= static_cast<int>(model->weights_per_rank.size())) {
        return nullptr;
    }
    return model->weights_per_rank[rank];
}

__export int64_t llaisysQwen2ModelTPInfer(
    struct LlaisysQwen2ModelTP* model,
    const int64_t* token_ids,
    size_t ntoken) {
    // Cast away const for the internal API
    return model->model->infer(const_cast<int64_t*>(token_ids), ntoken);
}

__export void llaisysQwen2ModelTPResetCache(struct LlaisysQwen2ModelTP* model) {
    model->model->resetCache();
}

__export int llaisysQwen2ModelTPGetWorldSize(struct LlaisysQwen2ModelTP* model) {
    return model->model->getWorldSize();
}

} // __C
