#include "llaisys/qwen2.h"

#include "llaisys_tensor.hpp"
#include "../models/qwen2/qwen2.hpp"

#include <cstdlib>

__C {
    struct LlaisysQwen2Model {
        llaisys::models::Qwen2Model* model;
        LlaisysQwen2Weights weights_c;
        size_t nlayer;
    };

    llaisysQwen2Model_t llaisysQwen2ModelCreate(
        const struct LlaisysQwen2Meta* meta,
        llaisysDeviceType_t device_type,
        int device_id) {

        // Convert C meta to C++ meta
        llaisys::models::Qwen2Meta cpp_meta;
        cpp_meta.nlayer = meta->nlayer;
        cpp_meta.hs = meta->hs;
        cpp_meta.nh = meta->nh;
        cpp_meta.nkvh = meta->nkvh;
        cpp_meta.dh = meta->dh;
        cpp_meta.di = meta->di;
        cpp_meta.maxseq = meta->maxseq;
        cpp_meta.voc = meta->voc;
        cpp_meta.epsilon = meta->epsilon;
        cpp_meta.theta = meta->theta;
        cpp_meta.end_token = meta->end_token;

        auto* handle = new LlaisysQwen2Model();
        handle->model = new llaisys::models::Qwen2Model(&cpp_meta, device_type, device_id);
        handle->nlayer = meta->nlayer;

        auto* cpp_weights = handle->model->weights();

        // Allocate arrays for per-layer weights
        size_t nlayer = meta->nlayer;
        handle->weights_c.attn_norm_w = new llaisysTensor_t[nlayer];
        handle->weights_c.attn_q_w = new llaisysTensor_t[nlayer];
        handle->weights_c.attn_q_b = new llaisysTensor_t[nlayer];
        handle->weights_c.attn_k_w = new llaisysTensor_t[nlayer];
        handle->weights_c.attn_k_b = new llaisysTensor_t[nlayer];
        handle->weights_c.attn_v_w = new llaisysTensor_t[nlayer];
        handle->weights_c.attn_v_b = new llaisysTensor_t[nlayer];
        handle->weights_c.attn_o_w = new llaisysTensor_t[nlayer];
        handle->weights_c.mlp_norm_w = new llaisysTensor_t[nlayer];
        handle->weights_c.mlp_gate_w = new llaisysTensor_t[nlayer];
        handle->weights_c.mlp_up_w = new llaisysTensor_t[nlayer];
        handle->weights_c.mlp_down_w = new llaisysTensor_t[nlayer];

        // Wrap embedding weights
        handle->weights_c.in_embed = new LlaisysTensor{cpp_weights->in_embed};
        handle->weights_c.out_embed = new LlaisysTensor{cpp_weights->out_embed};
        handle->weights_c.out_norm_w = new LlaisysTensor{cpp_weights->out_norm_w};

        // Wrap per-layer weights
        for (size_t i = 0; i < nlayer; i++) {
            handle->weights_c.attn_norm_w[i] = new LlaisysTensor{cpp_weights->attn_norm_w[i]};
            handle->weights_c.attn_q_w[i] = new LlaisysTensor{cpp_weights->attn_q_w[i]};
            handle->weights_c.attn_q_b[i] = new LlaisysTensor{cpp_weights->attn_q_b[i]};
            handle->weights_c.attn_k_w[i] = new LlaisysTensor{cpp_weights->attn_k_w[i]};
            handle->weights_c.attn_k_b[i] = new LlaisysTensor{cpp_weights->attn_k_b[i]};
            handle->weights_c.attn_v_w[i] = new LlaisysTensor{cpp_weights->attn_v_w[i]};
            handle->weights_c.attn_v_b[i] = new LlaisysTensor{cpp_weights->attn_v_b[i]};
            handle->weights_c.attn_o_w[i] = new LlaisysTensor{cpp_weights->attn_o_w[i]};
            handle->weights_c.mlp_norm_w[i] = new LlaisysTensor{cpp_weights->mlp_norm_w[i]};
            handle->weights_c.mlp_gate_w[i] = new LlaisysTensor{cpp_weights->mlp_gate_w[i]};
            handle->weights_c.mlp_up_w[i] = new LlaisysTensor{cpp_weights->mlp_up_w[i]};
            handle->weights_c.mlp_down_w[i] = new LlaisysTensor{cpp_weights->mlp_down_w[i]};
        }

        return handle;
    }

    void llaisysQwen2ModelDestroy(llaisysQwen2Model_t model) {
        if (!model) return;

        size_t nlayer = model->nlayer;

        // Delete wrapper tensors (not the underlying tensors, they're managed by C++)
        delete model->weights_c.in_embed;
        delete model->weights_c.out_embed;
        delete model->weights_c.out_norm_w;

        for (size_t i = 0; i < nlayer; i++) {
            delete model->weights_c.attn_norm_w[i];
            delete model->weights_c.attn_q_w[i];
            delete model->weights_c.attn_q_b[i];
            delete model->weights_c.attn_k_w[i];
            delete model->weights_c.attn_k_b[i];
            delete model->weights_c.attn_v_w[i];
            delete model->weights_c.attn_v_b[i];
            delete model->weights_c.attn_o_w[i];
            delete model->weights_c.mlp_norm_w[i];
            delete model->weights_c.mlp_gate_w[i];
            delete model->weights_c.mlp_up_w[i];
            delete model->weights_c.mlp_down_w[i];
        }

        delete[] model->weights_c.attn_norm_w;
        delete[] model->weights_c.attn_q_w;
        delete[] model->weights_c.attn_q_b;
        delete[] model->weights_c.attn_k_w;
        delete[] model->weights_c.attn_k_b;
        delete[] model->weights_c.attn_v_w;
        delete[] model->weights_c.attn_v_b;
        delete[] model->weights_c.attn_o_w;
        delete[] model->weights_c.mlp_norm_w;
        delete[] model->weights_c.mlp_gate_w;
        delete[] model->weights_c.mlp_up_w;
        delete[] model->weights_c.mlp_down_w;

        delete model->model;
        delete model;
    }

    struct LlaisysQwen2Weights* llaisysQwen2ModelWeights(llaisysQwen2Model_t model) {
        return &model->weights_c;
    }

    int64_t llaisysQwen2ModelInfer(
        llaisysQwen2Model_t model,
        int64_t* token_ids,
        size_t ntoken) {
        return model->model->infer(token_ids, ntoken);
    }

    void llaisysQwen2ModelResetCache(llaisysQwen2Model_t model) {
        model->model->resetCache();
    }
}
