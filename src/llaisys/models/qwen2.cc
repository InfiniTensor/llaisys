#include "llaisys/models/qwen2.h"
#include "../../models/qwen2/qwen2.hpp"
#include "../llaisys_tensor.hpp"

// Helper
llaisysTensor_t to_c(llaisys::tensor_t t) { return new LlaisysTensor{t}; }

__C {

struct LlaisysQwen2Model {
    llaisys::models::qwen2::Qwen2Model *model;
    struct LlaisysQwen2Weights c_weights;
    // Buffers to hold array pointers
    std::vector<llaisysTensor_t> p_attn_norm, p_attn_q_w, p_attn_q_b, p_attn_k_w, p_attn_k_b, p_attn_v_w, p_attn_v_b, p_attn_o_w;
    std::vector<llaisysTensor_t> p_mlp_norm, p_mlp_gate, p_mlp_up, p_mlp_down;
};

LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
    auto qwen = new llaisys::models::qwen2::Qwen2Model(meta, device, ndevice > 0 ? device_ids[0] : 0);
    auto wrapper = new LlaisysQwen2Model;
    wrapper->model = qwen;

    wrapper->c_weights.in_embed = to_c(qwen->in_embed);
    wrapper->c_weights.out_embed = to_c(qwen->out_embed);
    wrapper->c_weights.out_norm_w = to_c(qwen->out_norm_w);

    auto fill = [&](const std::vector<llaisys::tensor_t>& src, std::vector<llaisysTensor_t>& buf, llaisysTensor_t*& dst_ptr) {
        buf.resize(src.size());
        for(size_t i=0; i<src.size(); ++i) buf[i] = to_c(src[i]);
        dst_ptr = buf.data();
    };

    fill(qwen->attn_norm_w, wrapper->p_attn_norm, wrapper->c_weights.attn_norm_w);
    fill(qwen->attn_q_w, wrapper->p_attn_q_w, wrapper->c_weights.attn_q_w);
    fill(qwen->attn_q_b, wrapper->p_attn_q_b, wrapper->c_weights.attn_q_b);
    fill(qwen->attn_k_w, wrapper->p_attn_k_w, wrapper->c_weights.attn_k_w);
    fill(qwen->attn_k_b, wrapper->p_attn_k_b, wrapper->c_weights.attn_k_b);
    fill(qwen->attn_v_w, wrapper->p_attn_v_w, wrapper->c_weights.attn_v_w);
    fill(qwen->attn_v_b, wrapper->p_attn_v_b, wrapper->c_weights.attn_v_b);
    fill(qwen->attn_o_w, wrapper->p_attn_o_w, wrapper->c_weights.attn_o_w);
    fill(qwen->mlp_norm_w, wrapper->p_mlp_norm, wrapper->c_weights.mlp_norm_w);
    fill(qwen->mlp_gate_w, wrapper->p_mlp_gate, wrapper->c_weights.mlp_gate_w);
    fill(qwen->mlp_up_w, wrapper->p_mlp_up, wrapper->c_weights.mlp_up_w);
    fill(qwen->mlp_down_w, wrapper->p_mlp_down, wrapper->c_weights.mlp_down_w);

    return wrapper;
}

void llaisysQwen2ModelDestroy(LlaisysQwen2Model * model) {
    if (model) {
        auto free_t = [](llaisysTensor_t t) { if(t) delete t; };
        free_t(model->c_weights.in_embed);
        free_t(model->c_weights.out_embed);
        free_t(model->c_weights.out_norm_w);
        for(auto t : model->p_attn_norm) free_t(t);
        // ... free other vectors ...
        delete model->model;
        delete model;
    }
}

LlaisysQwen2Weights *llaisysQwen2ModelWeights(LlaisysQwen2Model * model) {
    return &model->c_weights;
}

int64_t llaisysQwen2ModelInfer(LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken) {
    return model->model->forward(token_ids, ntoken);
}

}