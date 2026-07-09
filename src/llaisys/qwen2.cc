#include "llaisys/models/qwen2.h"
#include "llaisys_tensor.hpp"
#include "../models/qwen2.hpp"

#include <deque>

__C {
    struct LlaisysQwen2Model {
        llaisys::models::Qwen2Model *model;
        LlaisysQwen2Weights c_weights;
        std::vector<llaisysTensor_t> attn_norm_w_ptrs;
        std::vector<llaisysTensor_t> attn_q_w_ptrs, attn_q_b_ptrs;
        std::vector<llaisysTensor_t> attn_k_w_ptrs, attn_k_b_ptrs;
        std::vector<llaisysTensor_t> attn_v_w_ptrs, attn_v_b_ptrs;
        std::vector<llaisysTensor_t> attn_o_w_ptrs;
        std::vector<llaisysTensor_t> mlp_norm_w_ptrs;
        std::vector<llaisysTensor_t> mlp_gate_w_ptrs, mlp_up_w_ptrs, mlp_down_w_ptrs;
        // Use deque to avoid pointer invalidation on push_back
        std::deque<LlaisysTensor> tensor_store;
    };

    struct LlaisysQwen2Model *llaisysQwen2ModelCreate(
        const LlaisysQwen2Meta *meta,
        llaisysDeviceType_t device,
        int *device_ids,
        int ndevice) {

        llaisys::models::Qwen2Config config;
        config.dtype = meta->dtype;
        config.nlayer = meta->nlayer;
        config.hs = meta->hs;
        config.nh = meta->nh;
        config.nkvh = meta->nkvh;
        config.dh = meta->dh;
        config.di = meta->di;
        config.maxseq = meta->maxseq;
        config.voc = meta->voc;
        config.epsilon = meta->epsilon;
        config.theta = meta->theta;
        config.end_token = meta->end_token;

        int device_id = (ndevice > 0) ? device_ids[0] : 0;

        auto *w = new LlaisysQwen2Model();
        w->model = new llaisys::models::Qwen2Model(config, device, device_id);

        auto &weights = w->model->weights();
        size_t nlayer = config.nlayer;
        size_t hs = config.hs, nh = config.nh, nkvh = config.nkvh;
        size_t dh = config.dh, di = config.di, voc = config.voc;
        auto dtype = config.dtype;

        auto wrap = [&](llaisys::tensor_t t) -> llaisysTensor_t {
            w->tensor_store.push_back(LlaisysTensor{t});
            return &w->tensor_store.back();
        };

        weights.in_embed = llaisys::Tensor::create({voc, hs}, dtype, device, device_id);
        weights.out_embed = llaisys::Tensor::create({voc, hs}, dtype, device, device_id);
        weights.out_norm_w = llaisys::Tensor::create({hs}, dtype, device, device_id);

        w->c_weights.in_embed = wrap(weights.in_embed);
        w->c_weights.out_embed = wrap(weights.out_embed);
        w->c_weights.out_norm_w = wrap(weights.out_norm_w);

        w->attn_norm_w_ptrs.resize(nlayer);
        w->attn_q_w_ptrs.resize(nlayer);
        w->attn_q_b_ptrs.resize(nlayer);
        w->attn_k_w_ptrs.resize(nlayer);
        w->attn_k_b_ptrs.resize(nlayer);
        w->attn_v_w_ptrs.resize(nlayer);
        w->attn_v_b_ptrs.resize(nlayer);
        w->attn_o_w_ptrs.resize(nlayer);
        w->mlp_norm_w_ptrs.resize(nlayer);
        w->mlp_gate_w_ptrs.resize(nlayer);
        w->mlp_up_w_ptrs.resize(nlayer);
        w->mlp_down_w_ptrs.resize(nlayer);

        for (size_t i = 0; i < nlayer; i++) {
            auto &lw = weights.layers[i];
            lw.attn_norm_w = llaisys::Tensor::create({hs}, dtype, device, device_id);
            lw.attn_q_w = llaisys::Tensor::create({nh * dh, hs}, dtype, device, device_id);
            lw.attn_q_b = llaisys::Tensor::create({nh * dh}, dtype, device, device_id);
            lw.attn_k_w = llaisys::Tensor::create({nkvh * dh, hs}, dtype, device, device_id);
            lw.attn_k_b = llaisys::Tensor::create({nkvh * dh}, dtype, device, device_id);
            lw.attn_v_w = llaisys::Tensor::create({nkvh * dh, hs}, dtype, device, device_id);
            lw.attn_v_b = llaisys::Tensor::create({nkvh * dh}, dtype, device, device_id);
            lw.attn_o_w = llaisys::Tensor::create({hs, nh * dh}, dtype, device, device_id);
            lw.mlp_norm_w = llaisys::Tensor::create({hs}, dtype, device, device_id);
            lw.mlp_gate_w = llaisys::Tensor::create({di, hs}, dtype, device, device_id);
            lw.mlp_up_w = llaisys::Tensor::create({di, hs}, dtype, device, device_id);
            lw.mlp_down_w = llaisys::Tensor::create({hs, di}, dtype, device, device_id);

            w->attn_norm_w_ptrs[i] = wrap(lw.attn_norm_w);
            w->attn_q_w_ptrs[i] = wrap(lw.attn_q_w);
            w->attn_q_b_ptrs[i] = wrap(lw.attn_q_b);
            w->attn_k_w_ptrs[i] = wrap(lw.attn_k_w);
            w->attn_k_b_ptrs[i] = wrap(lw.attn_k_b);
            w->attn_v_w_ptrs[i] = wrap(lw.attn_v_w);
            w->attn_v_b_ptrs[i] = wrap(lw.attn_v_b);
            w->attn_o_w_ptrs[i] = wrap(lw.attn_o_w);
            w->mlp_norm_w_ptrs[i] = wrap(lw.mlp_norm_w);
            w->mlp_gate_w_ptrs[i] = wrap(lw.mlp_gate_w);
            w->mlp_up_w_ptrs[i] = wrap(lw.mlp_up_w);
            w->mlp_down_w_ptrs[i] = wrap(lw.mlp_down_w);
        }

        w->c_weights.attn_norm_w = w->attn_norm_w_ptrs.data();
        w->c_weights.attn_q_w = w->attn_q_w_ptrs.data();
        w->c_weights.attn_q_b = w->attn_q_b_ptrs.data();
        w->c_weights.attn_k_w = w->attn_k_w_ptrs.data();
        w->c_weights.attn_k_b = w->attn_k_b_ptrs.data();
        w->c_weights.attn_v_w = w->attn_v_w_ptrs.data();
        w->c_weights.attn_v_b = w->attn_v_b_ptrs.data();
        w->c_weights.attn_o_w = w->attn_o_w_ptrs.data();
        w->c_weights.mlp_norm_w = w->mlp_norm_w_ptrs.data();
        w->c_weights.mlp_gate_w = w->mlp_gate_w_ptrs.data();
        w->c_weights.mlp_up_w = w->mlp_up_w_ptrs.data();
        w->c_weights.mlp_down_w = w->mlp_down_w_ptrs.data();

        return w;
    }

    void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
        if (model) {
            delete model->model;
            delete model;
        }
    }

    struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
        return &model->c_weights;
    }

    int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
        return model->model->infer(token_ids, ntoken);
    }

    int64_t llaisysQwen2ModelInferSample(struct LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken,
                                          float temperature, int top_k, float top_p) {
        return model->model->infer_sample(token_ids, ntoken, temperature, top_k, top_p);
    }

    void llaisysQwen2ModelResetKVCache(struct LlaisysQwen2Model *model) {
        model->model->reset_kvcache();
    }
}
