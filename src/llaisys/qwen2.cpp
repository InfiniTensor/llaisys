#include "llaisys/models/qwen2.h"
#include "../tensor/tensor.hpp"
#include "../utils.hpp"
#include "../ops/add/op.hpp"
#include "../ops/argmax/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"
#include "llaisys_tensor.hpp"

#include <vector>
#include <cstring>
#include <iostream>
#include <cmath>
#include <memory>
#include <cstdint>

using namespace llaisys;


// --- Linux 兼容性补丁 Start ---
#include <vector>
#include <string>
#include <cstring>  // 对应 memcpy, memset
#include <cmath>    // 对应 exp, sqrt, pow
#include <iostream> // 对应 printf, std::cout
#include <fstream>  // 对应文件读写
#include <algorithm> // 对应 std::max, std::min
// --- Linux 兼容性补丁 End ---
struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta;
    LlaisysQwen2Weights exported_weights;
    std::vector<tensor_t> k_cache, v_cache;
    tensor_t hidden_states, residual, attn_out, q, k, v, gate, up, mlp_out, logits;
    tensor_t input_ids, pos_ids, next_token, next_token_val;
    int64_t current_pos = 0;

    LlaisysQwen2Model(const LlaisysQwen2Meta& m) : meta(m) {
        // 分配指针数组
        auto alloc_arr = [&](size_t n) { return new llaisysTensor_t[n]; };
        exported_weights.attn_norm_w = alloc_arr(meta.nlayer);
        exported_weights.attn_q_w = alloc_arr(meta.nlayer);
        exported_weights.attn_q_b = alloc_arr(meta.nlayer);
        exported_weights.attn_k_w = alloc_arr(meta.nlayer);
        exported_weights.attn_k_b = alloc_arr(meta.nlayer);
        exported_weights.attn_v_w = alloc_arr(meta.nlayer);
        exported_weights.attn_v_b = alloc_arr(meta.nlayer);
        exported_weights.attn_o_w = alloc_arr(meta.nlayer);
        exported_weights.mlp_norm_w = alloc_arr(meta.nlayer);
        exported_weights.mlp_gate_w = alloc_arr(meta.nlayer);
        exported_weights.mlp_up_w = alloc_arr(meta.nlayer);
        exported_weights.mlp_down_w = alloc_arr(meta.nlayer);
    }

    ~LlaisysQwen2Model() {
        delete[] exported_weights.attn_norm_w; delete[] exported_weights.attn_q_w; delete[] exported_weights.attn_q_b;
        delete[] exported_weights.attn_k_w; delete[] exported_weights.attn_k_b; delete[] exported_weights.attn_v_w;
        delete[] exported_weights.attn_v_b; delete[] exported_weights.attn_o_w; delete[] exported_weights.mlp_norm_w;
        delete[] exported_weights.mlp_gate_w; delete[] exported_weights.mlp_up_w; delete[] exported_weights.mlp_down_w;
    }
};

extern "C" {

llaisysTensor_t create_wrapper(const std::vector<size_t>& shape) {
    auto cpp_tensor = Tensor::create(shape, LLAISYS_DTYPE_F32);
    // 初始化为 0 (防止某些层没有 Bias 时出现随机值)
    std::memset(cpp_tensor->data(), 0, cpp_tensor->numel() * sizeof(float));
    return new LlaisysTensor{cpp_tensor};
}

__export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
    auto model = new LlaisysQwen2Model(*meta);

    model->exported_weights.in_embed = create_wrapper({meta->voc, meta->hs});
    model->exported_weights.out_embed = create_wrapper({meta->voc, meta->hs});
    model->exported_weights.out_norm_w = create_wrapper({meta->hs});



    for (size_t i = 0; i < meta->nlayer; ++i) {   
        model->exported_weights.attn_norm_w[i] = create_wrapper({meta->hs});
        
        // QKV Weights
        model->exported_weights.attn_q_w[i] = create_wrapper({meta->nh * meta->dh, meta->hs});
        model->exported_weights.attn_k_w[i] = create_wrapper({meta->nkvh * meta->dh, meta->hs});
        model->exported_weights.attn_v_w[i] = create_wrapper({meta->nkvh * meta->dh, meta->hs});
        model->exported_weights.attn_o_w[i] = create_wrapper({meta->hs, meta->nh * meta->dh});

        // 🟢 修复1: 分配真实的 Bias 空间
        // Qwen2 QKV 有 Bias，大小等于输出维度
        model->exported_weights.attn_q_b[i] = create_wrapper({meta->nh * meta->dh});
        model->exported_weights.attn_k_b[i] = create_wrapper({meta->nkvh * meta->dh});
        model->exported_weights.attn_v_b[i] = create_wrapper({meta->nkvh * meta->dh});

        model->exported_weights.mlp_norm_w[i] = create_wrapper({meta->hs});
        model->exported_weights.mlp_gate_w[i] = create_wrapper({meta->di, meta->hs});
        model->exported_weights.mlp_up_w[i]   = create_wrapper({meta->di, meta->hs});
        model->exported_weights.mlp_down_w[i] = create_wrapper({meta->hs, meta->di});

        auto kc = Tensor::create({meta->maxseq, meta->nkvh, meta->dh}, LLAISYS_DTYPE_F32);
        auto vc = Tensor::create({meta->maxseq, meta->nkvh, meta->dh}, LLAISYS_DTYPE_F32);
        std::memset(kc->data(), 0, kc->numel() * sizeof(float));
        std::memset(vc->data(), 0, vc->numel() * sizeof(float));
        model->k_cache.push_back(kc);
        model->v_cache.push_back(vc);
    }

    model->hidden_states = Tensor::create({1, meta->hs}, LLAISYS_DTYPE_F32);
    model->residual      = Tensor::create({1, meta->hs}, LLAISYS_DTYPE_F32);
    model->attn_out      = Tensor::create({1, meta->hs}, LLAISYS_DTYPE_F32);
    model->q = Tensor::create({1, meta->nh, meta->dh}, LLAISYS_DTYPE_F32);
    model->k = Tensor::create({1, meta->nkvh, meta->dh}, LLAISYS_DTYPE_F32);
    model->v = Tensor::create({1, meta->nkvh, meta->dh}, LLAISYS_DTYPE_F32);
    model->gate    = Tensor::create({1, meta->di}, LLAISYS_DTYPE_F32);
    model->up      = Tensor::create({1, meta->di}, LLAISYS_DTYPE_F32);
    model->mlp_out = Tensor::create({1, meta->hs}, LLAISYS_DTYPE_F32);
    model->logits = Tensor::create({1, meta->voc}, LLAISYS_DTYPE_F32);
    
    model->input_ids = Tensor::create({1}, LLAISYS_DTYPE_I64);
    model->pos_ids   = Tensor::create({1}, LLAISYS_DTYPE_I64);
    model->next_token     = Tensor::create({1}, LLAISYS_DTYPE_I64);
    model->next_token_val = Tensor::create({1}, LLAISYS_DTYPE_F32);

    return model;
}

__export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model) {
    if (!model) return;
    auto free_w = [](llaisysTensor_t w) { if(w) delete w; };
    free_w(model->exported_weights.in_embed);
    free_w(model->exported_weights.out_embed);
    free_w(model->exported_weights.out_norm_w);
    for (size_t i = 0; i < model->meta.nlayer; ++i) {
        free_w(model->exported_weights.attn_norm_w[i]);
        free_w(model->exported_weights.attn_q_w[i]);
        free_w(model->exported_weights.attn_k_w[i]);
        free_w(model->exported_weights.attn_v_w[i]);
        free_w(model->exported_weights.attn_o_w[i]);
        free_w(model->exported_weights.attn_q_b[i]);
        free_w(model->exported_weights.attn_k_b[i]);
        free_w(model->exported_weights.attn_v_b[i]);
        free_w(model->exported_weights.mlp_norm_w[i]);
        free_w(model->exported_weights.mlp_gate_w[i]);
        free_w(model->exported_weights.mlp_up_w[i]);
        free_w(model->exported_weights.mlp_down_w[i]);
    }
    delete model;
}

__export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model) {
    return &model->exported_weights;
}

__export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken) {
    if (ntoken == 0) return 0;
    if (ntoken > 1) model->current_pos = 0;

    for (size_t i = 0; i < ntoken; ++i) {
        int64_t pos = model->current_pos;
        int64_t token = token_ids[i];

        reinterpret_cast<int64_t*>(model->input_ids->data())[0] = token;
        llaisys::ops::embedding(model->hidden_states, model->input_ids, model->exported_weights.in_embed->tensor);
        
        reinterpret_cast<int64_t*>(model->pos_ids->data())[0] = pos;

        for (size_t l = 0; l < model->meta.nlayer; ++l) {
            std::memcpy(model->residual->data(), model->hidden_states->data(), model->meta.hs * sizeof(float));
            llaisys::ops::rms_norm(model->attn_out, model->hidden_states, model->exported_weights.attn_norm_w[l]->tensor, model->meta.epsilon);

            // 🟢 修复2: 传入 Bias (QKV)
            llaisys::ops::linear(model->q, model->attn_out, model->exported_weights.attn_q_w[l]->tensor, model->exported_weights.attn_q_b[l]->tensor);
            llaisys::ops::linear(model->k, model->attn_out, model->exported_weights.attn_k_w[l]->tensor, model->exported_weights.attn_k_b[l]->tensor);
            llaisys::ops::linear(model->v, model->attn_out, model->exported_weights.attn_v_w[l]->tensor, model->exported_weights.attn_v_b[l]->tensor);

            llaisys::ops::rope(model->q, model->q, model->pos_ids, model->meta.theta);
            llaisys::ops::rope(model->k, model->k, model->pos_ids, model->meta.theta);

            size_t kv_bytes = model->meta.nkvh * model->meta.dh * sizeof(float);
            float* kc_ptr = reinterpret_cast<float*>(model->k_cache[l]->data()) + pos * model->meta.nkvh * model->meta.dh;
            float* vc_ptr = reinterpret_cast<float*>(model->v_cache[l]->data()) + pos * model->meta.nkvh * model->meta.dh;
            std::memcpy(kc_ptr, model->k->data(), kv_bytes);
            std::memcpy(vc_ptr, model->v->data(), kv_bytes);

            auto k_view = model->k_cache[l]->slice(0, 0, pos + 1);
            auto v_view = model->v_cache[l]->slice(0, 0, pos + 1);

            float scale = 1.0f / std::sqrt((float)model->meta.dh);
            llaisys::ops::self_attention(model->attn_out, model->q, k_view, v_view, scale);

            // O Proj 通常无 Bias
            llaisys::ops::linear(model->hidden_states, model->attn_out, model->exported_weights.attn_o_w[l]->tensor, nullptr);
            llaisys::ops::add(model->hidden_states, model->residual, model->hidden_states);

            std::memcpy(model->residual->data(), model->hidden_states->data(), model->meta.hs * sizeof(float));
            llaisys::ops::rms_norm(model->attn_out, model->hidden_states, model->exported_weights.mlp_norm_w[l]->tensor, model->meta.epsilon);
            
            // MLP 通常无 Bias
            llaisys::ops::linear(model->gate, model->attn_out, model->exported_weights.mlp_gate_w[l]->tensor, nullptr);
            llaisys::ops::linear(model->up,   model->attn_out, model->exported_weights.mlp_up_w[l]->tensor,   nullptr);
            llaisys::ops::swiglu(model->up, model->gate, model->up);
            llaisys::ops::linear(model->mlp_out, model->up, model->exported_weights.mlp_down_w[l]->tensor, nullptr);
            llaisys::ops::add(model->hidden_states, model->residual, model->mlp_out);
        }

        llaisys::ops::rms_norm(model->hidden_states, model->hidden_states, model->exported_weights.out_norm_w->tensor, model->meta.epsilon);
        llaisys::ops::linear(model->logits, model->hidden_states, model->exported_weights.out_embed->tensor, nullptr);

        model->current_pos++;
    }

    llaisys::ops::argmax(model->next_token, model->next_token_val, model->logits);
    return reinterpret_cast<int64_t*>(model->next_token->data())[0];
}

} // extern "C"