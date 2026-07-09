#include "llaisys/models/qwen2.h"
#include "../../models/qwen2/qwen2.hpp"
#include "../../core/llaisys_core.hpp"
#include "../llaisys_tensor.hpp"

using namespace llaisys;

struct LlaisysQwen2Model {
    std::unique_ptr<models::Qwen2Model> model;
};

__export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(
    const LlaisysQwen2Meta *meta,
    llaisysDeviceType_t device,
    int *device_ids,
    int ndevice) {

    // Convert meta to config
    models::Qwen2Config config;
    config.dtype = meta->dtype;
    config.n_layers = meta->nlayer;
    config.hidden_size = meta->hs;
    config.n_heads = meta->nh;
    config.n_kv_heads = meta->nkvh;
    config.head_dim = meta->dh;
    config.intermediate_size = meta->di;
    config.max_seq_len = meta->maxseq;
    config.vocab_size = meta->voc;
    config.rms_norm_eps = meta->epsilon;
    config.rope_theta = meta->theta;
    config.eos_token_id = meta->end_token;

    // For now, only support single device
    int device_id = (ndevice > 0 && device_ids != nullptr) ? device_ids[0] : 0;

    auto model_wrapper = new LlaisysQwen2Model();
    model_wrapper->model = std::make_unique<models::Qwen2Model>(config, device, device_id);

    return model_wrapper;
}

__export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
    delete model;
}

__export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
    if (!model || !model->model) {
        return nullptr;
    }

    auto &cpp_weights = model->model->weights();
    const auto &config = model->model->config();

    // Allocate C struct
    auto weights = new LlaisysQwen2Weights();

    // Wrap tensor_t (shared_ptr) in LlaisysTensor struct
    weights->in_embed = new LlaisysTensor{cpp_weights.embed_tokens};
    weights->out_embed = new LlaisysTensor{cpp_weights.lm_head};
    weights->out_norm_w = new LlaisysTensor{cpp_weights.norm_weight};

    // Allocate arrays for per-layer weights
    size_t n_layers = config.n_layers;

    weights->attn_norm_w = new llaisysTensor_t[n_layers];
    weights->attn_q_w = new llaisysTensor_t[n_layers];
    weights->attn_q_b = new llaisysTensor_t[n_layers];
    weights->attn_k_w = new llaisysTensor_t[n_layers];
    weights->attn_k_b = new llaisysTensor_t[n_layers];
    weights->attn_v_w = new llaisysTensor_t[n_layers];
    weights->attn_v_b = new llaisysTensor_t[n_layers];
    weights->attn_o_w = new llaisysTensor_t[n_layers];
    weights->mlp_norm_w = new llaisysTensor_t[n_layers];
    weights->mlp_gate_w = new llaisysTensor_t[n_layers];
    weights->mlp_up_w = new llaisysTensor_t[n_layers];
    weights->mlp_down_w = new llaisysTensor_t[n_layers];

    for (size_t i = 0; i < n_layers; i++) {
        weights->attn_norm_w[i] = new LlaisysTensor{cpp_weights.attn_norm_weight[i]};
        weights->attn_q_w[i] = new LlaisysTensor{cpp_weights.attn_q_weight[i]};
        weights->attn_q_b[i] = new LlaisysTensor{cpp_weights.attn_q_bias[i]};
        weights->attn_k_w[i] = new LlaisysTensor{cpp_weights.attn_k_weight[i]};
        weights->attn_k_b[i] = new LlaisysTensor{cpp_weights.attn_k_bias[i]};
        weights->attn_v_w[i] = new LlaisysTensor{cpp_weights.attn_v_weight[i]};
        weights->attn_v_b[i] = new LlaisysTensor{cpp_weights.attn_v_bias[i]};
        weights->attn_o_w[i] = new LlaisysTensor{cpp_weights.attn_o_weight[i]};
        weights->mlp_norm_w[i] = new LlaisysTensor{cpp_weights.mlp_norm_weight[i]};
        weights->mlp_gate_w[i] = new LlaisysTensor{cpp_weights.mlp_gate_weight[i]};
        weights->mlp_up_w[i] = new LlaisysTensor{cpp_weights.mlp_up_weight[i]};
        weights->mlp_down_w[i] = new LlaisysTensor{cpp_weights.mlp_down_weight[i]};
    }

    return weights;
}

__export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
    if (!model || !model->model || !token_ids || ntoken == 0) {
        return -1;
    }

    std::vector<int64_t> input_ids(token_ids, token_ids + ntoken);
    return model->model->generate_next_token(input_ids);
}
