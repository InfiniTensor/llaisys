#include "llaisys/models/qwen2.h"

#include "llaisys_tensor.hpp"
#include "../models/qwen2/qwen2.hpp"

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
    
    // Create config from meta
    llaisys::models::Qwen2Config config;
    config.dtype = meta->dtype;
    config.num_layers = meta->nlayer;
    config.hidden_size = meta->hs;
    config.num_heads = meta->nh;
    config.num_kv_heads = meta->nkvh;
    config.head_dim = meta->dh;
    config.intermediate_size = meta->di;
    config.max_seq_len = meta->maxseq;
    config.vocab_size = meta->voc;
    config.rms_norm_eps = meta->epsilon;
    config.rope_theta = meta->theta;
    config.eos_token_id = meta->end_token;
    
    // For now, use first device
    int device_id = (ndevice > 0 && device_ids != nullptr) ? device_ids[0] : 0;
    
    auto* wrapper = new LlaisysQwen2Model();
    wrapper->model = new llaisys::models::Qwen2Model(config, device, device_id);
    
    // Set up weight pointers
    auto& model = *wrapper->model;
    wrapper->weights.in_embed = new LlaisysTensor{model.embedTokens()};
    wrapper->weights.out_embed = new LlaisysTensor{model.lmHead()};
    wrapper->weights.out_norm_w = new LlaisysTensor{model.normWeight()};
    
    size_t nlayer = config.num_layers;
    wrapper->weights.attn_norm_w = new llaisysTensor_t[nlayer];
    wrapper->weights.attn_q_w = new llaisysTensor_t[nlayer];
    wrapper->weights.attn_q_b = new llaisysTensor_t[nlayer];
    wrapper->weights.attn_k_w = new llaisysTensor_t[nlayer];
    wrapper->weights.attn_k_b = new llaisysTensor_t[nlayer];
    wrapper->weights.attn_v_w = new llaisysTensor_t[nlayer];
    wrapper->weights.attn_v_b = new llaisysTensor_t[nlayer];
    wrapper->weights.attn_o_w = new llaisysTensor_t[nlayer];
    wrapper->weights.mlp_norm_w = new llaisysTensor_t[nlayer];
    wrapper->weights.mlp_gate_w = new llaisysTensor_t[nlayer];
    wrapper->weights.mlp_up_w = new llaisysTensor_t[nlayer];
    wrapper->weights.mlp_down_w = new llaisysTensor_t[nlayer];
    
    for (size_t i = 0; i < nlayer; ++i) {
        wrapper->weights.attn_norm_w[i] = new LlaisysTensor{model.inputLayernormWeight(i)};
        wrapper->weights.attn_q_w[i] = new LlaisysTensor{model.qProjWeight(i)};
        wrapper->weights.attn_q_b[i] = new LlaisysTensor{model.qProjBias(i)};
        wrapper->weights.attn_k_w[i] = new LlaisysTensor{model.kProjWeight(i)};
        wrapper->weights.attn_k_b[i] = new LlaisysTensor{model.kProjBias(i)};
        wrapper->weights.attn_v_w[i] = new LlaisysTensor{model.vProjWeight(i)};
        wrapper->weights.attn_v_b[i] = new LlaisysTensor{model.vProjBias(i)};
        wrapper->weights.attn_o_w[i] = new LlaisysTensor{model.oProjWeight(i)};
        wrapper->weights.mlp_norm_w[i] = new LlaisysTensor{model.postAttnLayernormWeight(i)};
        wrapper->weights.mlp_gate_w[i] = new LlaisysTensor{model.gateProjWeight(i)};
        wrapper->weights.mlp_up_w[i] = new LlaisysTensor{model.upProjWeight(i)};
        wrapper->weights.mlp_down_w[i] = new LlaisysTensor{model.downProjWeight(i)};
    }
    
    return wrapper;
}

__export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model* model) {
    if (model == nullptr) return;
    
    size_t nlayer = model->model->config().num_layers;
    
    // Delete weight wrappers
    delete model->weights.in_embed;
    delete model->weights.out_embed;
    delete model->weights.out_norm_w;
    
    for (size_t i = 0; i < nlayer; ++i) {
        delete model->weights.attn_norm_w[i];
        delete model->weights.attn_q_w[i];
        delete model->weights.attn_q_b[i];
        delete model->weights.attn_k_w[i];
        delete model->weights.attn_k_b[i];
        delete model->weights.attn_v_w[i];
        delete model->weights.attn_v_b[i];
        delete model->weights.attn_o_w[i];
        delete model->weights.mlp_norm_w[i];
        delete model->weights.mlp_gate_w[i];
        delete model->weights.mlp_up_w[i];
        delete model->weights.mlp_down_w[i];
    }
    
    delete[] model->weights.attn_norm_w;
    delete[] model->weights.attn_q_w;
    delete[] model->weights.attn_q_b;
    delete[] model->weights.attn_k_w;
    delete[] model->weights.attn_k_b;
    delete[] model->weights.attn_v_w;
    delete[] model->weights.attn_v_b;
    delete[] model->weights.attn_o_w;
    delete[] model->weights.mlp_norm_w;
    delete[] model->weights.mlp_gate_w;
    delete[] model->weights.mlp_up_w;
    delete[] model->weights.mlp_down_w;
    
    delete model->model;
    delete model;
}

__export struct LlaisysQwen2Weights* llaisysQwen2ModelWeights(struct LlaisysQwen2Model* model) {
    return &model->weights;
}

__export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model* model, int64_t* token_ids, size_t ntoken) {
    return model->model->infer(token_ids, ntoken);
}

}
