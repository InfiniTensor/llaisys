#include "llaisys/models/qwen2.h"

#include "llaisys_tensor.hpp"
#include "../models/qwen2/model.hpp"

#include <vector>
#include <memory>

// C++ Model 的包装结构
struct LlaisysQwen2Model {
    std::unique_ptr<llaisys::models::qwen2::Model> model;
    std::unique_ptr<LlaisysQwen2Weights> c_weights;  // C 结构的权重，由 Python 设置
};

// 同步权重从 C 结构到 C++ 模型
static void sync_weights(struct LlaisysQwen2Model *model) {
    if (!model->c_weights) return;
    
    auto& weights = model->model->weights();
    size_t nlayer = model->model->meta().nlayer;
    
    if (model->c_weights->in_embed) {
        weights.in_embed = model->c_weights->in_embed->tensor;
    }
    if (model->c_weights->out_embed) {
        weights.out_embed = model->c_weights->out_embed->tensor;
    }
    if (model->c_weights->out_norm_w) {
        weights.out_norm_w = model->c_weights->out_norm_w->tensor;
    }
    for (size_t i = 0; i < nlayer; ++i) {
        if (model->c_weights->attn_norm_w[i]) {
            weights.attn_norm_w[i] = model->c_weights->attn_norm_w[i]->tensor;
        }
        if (model->c_weights->attn_q_w[i]) {
            weights.attn_q_w[i] = model->c_weights->attn_q_w[i]->tensor;
        }
        if (model->c_weights->attn_q_b[i]) {
            weights.attn_q_b[i] = model->c_weights->attn_q_b[i]->tensor;
        }
        if (model->c_weights->attn_k_w[i]) {
            weights.attn_k_w[i] = model->c_weights->attn_k_w[i]->tensor;
        }
        if (model->c_weights->attn_k_b[i]) {
            weights.attn_k_b[i] = model->c_weights->attn_k_b[i]->tensor;
        }
        if (model->c_weights->attn_v_w[i]) {
            weights.attn_v_w[i] = model->c_weights->attn_v_w[i]->tensor;
        }
        if (model->c_weights->attn_v_b[i]) {
            weights.attn_v_b[i] = model->c_weights->attn_v_b[i]->tensor;
        }
        if (model->c_weights->attn_o_w[i]) {
            weights.attn_o_w[i] = model->c_weights->attn_o_w[i]->tensor;
        }
        if (model->c_weights->mlp_norm_w[i]) {
            weights.mlp_norm_w[i] = model->c_weights->mlp_norm_w[i]->tensor;
        }
        if (model->c_weights->mlp_gate_w[i]) {
            weights.mlp_gate_w[i] = model->c_weights->mlp_gate_w[i]->tensor;
        }
        if (model->c_weights->mlp_up_w[i]) {
            weights.mlp_up_w[i] = model->c_weights->mlp_up_w[i]->tensor;
        }
        if (model->c_weights->mlp_down_w[i]) {
            weights.mlp_down_w[i] = model->c_weights->mlp_down_w[i]->tensor;
        }
    }
}

__C {
    struct LlaisysQwen2Model *llaisysQwen2ModelCreate(
        const LlaisysQwen2Meta *meta,
        llaisysDeviceType_t device,
        int *device_ids,
        int ndevice) {
        
        llaisys::models::qwen2::ModelMeta cpp_meta;
        cpp_meta.dtype = meta->dtype;
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
        
        int device_id = (ndevice > 0 && device_ids) ? device_ids[0] : 0;
        
        auto model = std::make_unique<llaisys::models::qwen2::Model>(cpp_meta, device, device_id);
        
        return new LlaisysQwen2Model{std::move(model)};
    }
    
    void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
        delete model;
    }
    
    struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
        // 返回模型权重的引用，Python 侧可以设置这些指针
        // 如果还没有创建，则创建并初始化
        if (!model->c_weights) {
            size_t nlayer = model->model->meta().nlayer;
            model->c_weights = std::make_unique<LlaisysQwen2Weights>();
            
            // 初始化指针为 nullptr，由 Python 侧设置
            model->c_weights->in_embed = nullptr;
            model->c_weights->out_embed = nullptr;
            model->c_weights->out_norm_w = nullptr;
            
            // 为每层权重分配数组
            model->c_weights->attn_norm_w = new LlaisysTensor*[nlayer];
            model->c_weights->attn_q_w = new LlaisysTensor*[nlayer];
            model->c_weights->attn_q_b = new LlaisysTensor*[nlayer];
            model->c_weights->attn_k_w = new LlaisysTensor*[nlayer];
            model->c_weights->attn_k_b = new LlaisysTensor*[nlayer];
            model->c_weights->attn_v_w = new LlaisysTensor*[nlayer];
            model->c_weights->attn_v_b = new LlaisysTensor*[nlayer];
            model->c_weights->attn_o_w = new LlaisysTensor*[nlayer];
            model->c_weights->mlp_norm_w = new LlaisysTensor*[nlayer];
            model->c_weights->mlp_gate_w = new LlaisysTensor*[nlayer];
            model->c_weights->mlp_up_w = new LlaisysTensor*[nlayer];
            model->c_weights->mlp_down_w = new LlaisysTensor*[nlayer];
            
            // 初始化为 nullptr
            for (size_t i = 0; i < nlayer; ++i) {
                model->c_weights->attn_norm_w[i] = nullptr;
                model->c_weights->attn_q_w[i] = nullptr;
                model->c_weights->attn_q_b[i] = nullptr;
                model->c_weights->attn_k_w[i] = nullptr;
                model->c_weights->attn_k_b[i] = nullptr;
                model->c_weights->attn_v_w[i] = nullptr;
                model->c_weights->attn_v_b[i] = nullptr;
                model->c_weights->attn_o_w[i] = nullptr;
                model->c_weights->mlp_norm_w[i] = nullptr;
                model->c_weights->mlp_gate_w[i] = nullptr;
                model->c_weights->mlp_up_w[i] = nullptr;
                model->c_weights->mlp_down_w[i] = nullptr;
            }
        }
        
        // 每次调用时同步权重（确保权重是最新的）
        sync_weights(model);
        
        return model->c_weights.get();
    }
    
    void llaisysQwen2ModelResetCache(struct LlaisysQwen2Model *model) {
        model->model->reset_cache();
    }
    
    int64_t llaisysQwen2ModelInfer(
        struct LlaisysQwen2Model *model,
        int64_t *token_ids,
        size_t ntoken) {
        
        // 允许 Python 在任意时刻更新 c_weights 指针：
        // 推理前再同步一次，避免"先拿到 weights 指针 -> Python 填充 -> 没再调用 Weights()"导致的未同步问题。
        sync_weights(model);
        return model->model->infer(token_ids, ntoken);
    }

    int64_t llaisysQwen2ModelInferSample(
        struct LlaisysQwen2Model *model,
        int64_t *token_ids,
        size_t ntoken,
        int top_k,
        float top_p,
        float temperature) {

        sync_weights(model);
        return model->model->infer(token_ids, ntoken, top_k, top_p, temperature);
    }
}
