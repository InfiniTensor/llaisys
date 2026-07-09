#include "llaisys/models/qwen2.h"
#include "../tensor/tensor.hpp"
#include "../utils.hpp"

// 引用具体算子
#include "../ops/embedding/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"
#include "../ops/add/op.hpp"
// 【新增】：引入采样器算子！
#include "../ops/sampler/cpu/sampler_cpu.hpp"

#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <iostream>

using namespace llaisys;

// 类型桥接
struct LlaisysTensor { llaisys::tensor_t tensor; };
llaisysTensor_t wrap(tensor_t t) { return new LlaisysTensor{t}; }
tensor_t unwrap(llaisysTensor_t t) { return ((LlaisysTensor*)t)->tensor; }

class Qwen2Impl {
public:
    LlaisysQwen2Meta meta;
    LlaisysQwen2Weights weights;
    size_t current_pos;

    std::vector<tensor_t> k_cache;
    std::vector<tensor_t> v_cache;

    // Buffers
    tensor_t buf_input_ids, buf_pos_ids;
    tensor_t buf_hidden, buf_residual;
    
    // QKV 相关 Buffer
    tensor_t buf_q, buf_k, buf_v;
    tensor_t buf_att_out; 
    
    tensor_t buf_att_proj;
    tensor_t buf_gate, buf_up, buf_ffn_inter, buf_down;
    tensor_t buf_last_hidden, buf_logits;

public:
    Qwen2Impl(const LlaisysQwen2Meta* meta_in) {
        this->meta = *meta_in;
        this->current_pos = 0;
        alloc_weight_structs();
        init_weight_tensors();
        init_kv_cache();
        init_buffers();
    }

    ~Qwen2Impl() {
        free_layer_weights(weights.attn_norm_w);
        free_layer_weights(weights.attn_q_w); free_layer_weights(weights.attn_q_b);
        free_layer_weights(weights.attn_k_w); free_layer_weights(weights.attn_k_b);
        free_layer_weights(weights.attn_v_w); free_layer_weights(weights.attn_v_b);
        free_layer_weights(weights.attn_o_w);
        free_layer_weights(weights.mlp_norm_w);
        free_layer_weights(weights.mlp_gate_w);
        free_layer_weights(weights.mlp_up_w);
        free_layer_weights(weights.mlp_down_w);
        delete (LlaisysTensor*)weights.in_embed;
        delete (LlaisysTensor*)weights.out_embed;
        delete (LlaisysTensor*)weights.out_norm_w;
    }

private:
    void free_layer_weights(llaisysTensor_t* array) {
        if (!array) return;
        for (size_t i = 0; i < meta.nlayer; i++) if (array[i]) delete (LlaisysTensor*)array[i];
        delete[] array;
    }

    llaisysTensor_t create_w(const std::vector<size_t>& shape) {
        return wrap(Tensor::create(shape, LLAISYS_DTYPE_F32, LLAISYS_DEVICE_CPU));
    }
    tensor_t create_b(const std::vector<size_t>& shape, llaisysDataType_t dtype = LLAISYS_DTYPE_F32) {
        return Tensor::create(shape, dtype, LLAISYS_DEVICE_CPU);
    }

    void alloc_weight_structs() {
        size_t n = meta.nlayer;
        weights.attn_norm_w = new llaisysTensor_t[n];
        weights.attn_q_w = new llaisysTensor_t[n]; weights.attn_q_b = new llaisysTensor_t[n];
        weights.attn_k_w = new llaisysTensor_t[n]; weights.attn_k_b = new llaisysTensor_t[n];
        weights.attn_v_w = new llaisysTensor_t[n]; weights.attn_v_b = new llaisysTensor_t[n];
        weights.attn_o_w = new llaisysTensor_t[n];
        weights.mlp_norm_w = new llaisysTensor_t[n];
        weights.mlp_gate_w = new llaisysTensor_t[n];
        weights.mlp_up_w = new llaisysTensor_t[n];
        weights.mlp_down_w = new llaisysTensor_t[n];
    }

    void init_weight_tensors() {
        weights.in_embed = create_w({meta.voc, meta.hs});
        weights.out_embed = create_w({meta.voc, meta.hs});
        weights.out_norm_w = create_w({meta.hs});
        size_t qs = meta.nh * meta.dh;
        size_t kvs = meta.nkvh * meta.dh;
        for (size_t i = 0; i < meta.nlayer; i++) {
            weights.attn_norm_w[i] = create_w({meta.hs});
            weights.attn_q_w[i] = create_w({qs, meta.hs}); weights.attn_q_b[i] = create_w({qs});
            weights.attn_k_w[i] = create_w({kvs, meta.hs}); weights.attn_k_b[i] = create_w({kvs});
            weights.attn_v_w[i] = create_w({kvs, meta.hs}); weights.attn_v_b[i] = create_w({kvs});
            weights.attn_o_w[i] = create_w({meta.hs, qs});
            weights.mlp_norm_w[i] = create_w({meta.hs});
            weights.mlp_gate_w[i] = create_w({meta.di, meta.hs});
            weights.mlp_up_w[i] = create_w({meta.di, meta.hs});
            weights.mlp_down_w[i] = create_w({meta.hs, meta.di});
        }
    }

    void init_kv_cache() {
        std::vector<size_t> shape = {meta.maxseq, meta.nkvh, meta.dh};
        for (size_t i = 0; i < meta.nlayer; i++) {
            k_cache.push_back(create_b(shape));
            v_cache.push_back(create_b(shape));
        }
    }

    void init_buffers() {
        size_t s = meta.maxseq;
        size_t dim_q = meta.nh * meta.dh;
        size_t dim_kv = meta.nkvh * meta.dh;

        buf_input_ids = create_b({s}, LLAISYS_DTYPE_I64);
        buf_pos_ids   = create_b({s}, LLAISYS_DTYPE_I64);
        buf_hidden    = create_b({s, meta.hs});
        buf_residual  = create_b({s, meta.hs});

        buf_q = create_b({s, dim_q});  
        buf_k = create_b({s, dim_kv}); 
        buf_v = create_b({s, dim_kv}); 
        
        buf_att_out   = create_b({s, dim_q}); 
        buf_att_proj  = create_b({s, meta.hs});
        
        buf_gate      = create_b({s, meta.di});
        buf_up        = create_b({s, meta.di});
        buf_ffn_inter = create_b({s, meta.di});
        buf_down      = create_b({s, meta.hs});
        
        buf_last_hidden = create_b({1, meta.hs});
        buf_logits      = create_b({1, meta.voc});
    }

public:
    // 【修改】：接收采样超参数 temperature, top_p, top_k
    int64_t infer(int64_t* token_ids, size_t ntoken, float temperature, float top_p, size_t top_k) {
        size_t seq_len = ntoken;
        size_t start_pos = this->current_pos;
        size_t total_len = start_pos + seq_len;

        // === Step 1: Slice Views ===
        tensor_t cur_input_ids = buf_input_ids->slice(0, 0, seq_len);
        tensor_t cur_pos_ids   = buf_pos_ids->slice(0, 0, seq_len);
        tensor_t cur_hidden    = buf_hidden->slice(0, 0, seq_len);
        tensor_t cur_residual  = buf_residual->slice(0, 0, seq_len);

        tensor_t cur_q = buf_q->slice(0, 0, seq_len); 
        tensor_t cur_k = buf_k->slice(0, 0, seq_len); 
        tensor_t cur_v = buf_v->slice(0, 0, seq_len); 
        
        tensor_t cur_att_out_flat = buf_att_out->slice(0, 0, seq_len); 
        tensor_t cur_att_proj     = buf_att_proj->slice(0, 0, seq_len);

        tensor_t cur_gate      = buf_gate->slice(0, 0, seq_len);
        tensor_t cur_up        = buf_up->slice(0, 0, seq_len);
        tensor_t cur_ffn_inter = buf_ffn_inter->slice(0, 0, seq_len);
        tensor_t cur_down      = buf_down->slice(0, 0, seq_len);

        // === Step 2: Inference Loop ===
        std::memcpy(cur_input_ids->data(), token_ids, seq_len * sizeof(int64_t));
        int64_t* pos_ptr = (int64_t*)cur_pos_ids->data();
        for (size_t i = 0; i < seq_len; ++i) pos_ptr[i] = start_pos + i;

        ops::embedding(cur_hidden, cur_input_ids, unwrap(weights.in_embed));

        for (size_t i = 0; i < meta.nlayer; i++) {
            size_t bytes_hidden = cur_hidden->numel() * sizeof(float);
            std::memcpy(cur_residual->data(), cur_hidden->data(), bytes_hidden);
            
            ops::rms_norm(cur_hidden, cur_hidden, unwrap(weights.attn_norm_w[i]), meta.epsilon);
            
            ops::linear(cur_q, cur_hidden, unwrap(weights.attn_q_w[i]), unwrap(weights.attn_q_b[i]));
            ops::linear(cur_k, cur_hidden, unwrap(weights.attn_k_w[i]), unwrap(weights.attn_k_b[i]));
            ops::linear(cur_v, cur_hidden, unwrap(weights.attn_v_w[i]), unwrap(weights.attn_v_b[i]));
            
            tensor_t q_3d = cur_q->reshape({seq_len, meta.nh, meta.dh});
            tensor_t k_3d = cur_k->reshape({seq_len, meta.nkvh, meta.dh});
            
            ops::rope(q_3d, q_3d, cur_pos_ids, meta.theta);
            ops::rope(k_3d, k_3d, cur_pos_ids, meta.theta);
            
            tensor_t kc = k_cache[i];
            tensor_t vc = v_cache[i];
            size_t bytes_copy = seq_len * meta.nkvh * meta.dh * sizeof(float);
            size_t offset = start_pos * meta.nkvh * meta.dh * sizeof(float);
            std::memcpy(kc->data() + offset, cur_k->data(), bytes_copy);
            std::memcpy(vc->data() + offset, cur_v->data(), bytes_copy);
            
            tensor_t kc_view = kc->slice(0, 0, total_len)->reshape({total_len, meta.nkvh, meta.dh});
            tensor_t vc_view = vc->slice(0, 0, total_len)->reshape({total_len, meta.nkvh, meta.dh});
            tensor_t att_out_3d = cur_att_out_flat->reshape({seq_len, meta.nh, meta.dh});
            tensor_t v_3d = cur_v->reshape({seq_len, meta.nkvh, meta.dh}); 
            
            float scale = 1.0f / std::sqrt((float)meta.dh);
            ops::self_attention(att_out_3d, q_3d, kc_view, vc_view, scale);
            
            ops::linear(cur_att_proj, cur_att_out_flat, unwrap(weights.attn_o_w[i]), nullptr);
            ops::add(cur_hidden, cur_att_proj, cur_residual);
            
            std::memcpy(cur_residual->data(), cur_hidden->data(), bytes_hidden);
            
            ops::rms_norm(cur_hidden, cur_hidden, unwrap(weights.mlp_norm_w[i]), meta.epsilon);
            
            ops::linear(cur_gate, cur_hidden, unwrap(weights.mlp_gate_w[i]), nullptr);
            ops::linear(cur_up,   cur_hidden, unwrap(weights.mlp_up_w[i]),   nullptr);
            ops::swiglu(cur_ffn_inter, cur_gate, cur_up);
            ops::linear(cur_down, cur_ffn_inter, unwrap(weights.mlp_down_w[i]), nullptr);
            
            ops::add(cur_hidden, cur_down, cur_residual);
        }

        ops::rms_norm(cur_hidden, cur_hidden, unwrap(weights.out_norm_w), meta.epsilon);
        
        std::byte* last_row = cur_hidden->data() + (seq_len - 1) * meta.hs * sizeof(float);
        std::memcpy(buf_last_hidden->data(), last_row, meta.hs * sizeof(float));
        ops::linear(buf_logits, buf_last_hidden, unwrap(weights.out_embed), nullptr);

        // ==========================================
        // 【核心修改】：删除了 Argmax，调用强大的 Sampler
        // ==========================================
        float* logits = (float*)buf_logits->data();
        int64_t next_token = ops::cpu::sample(logits, meta.voc, temperature, top_p, top_k);

        this->current_pos += seq_len;
        return next_token;
    }
};

extern "C" {
__export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
    return (struct LlaisysQwen2Model *) new Qwen2Impl(meta);
}
__export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model) {
    if (model) delete (Qwen2Impl*)model;
}
__export LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model) {
    if (!model) return nullptr;
    return &((Qwen2Impl*)model)->weights;
}
// 【修改】：C-API 签名增加采样参数
__export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken, float temperature, float top_p, size_t top_k) {
    if (!model) return -1;
    return ((Qwen2Impl*)model)->infer(token_ids, ntoken, temperature, top_p, top_k);
}
__export void* llaisysTensorData(void* t) {
    if (!t) return nullptr;
    return ((LlaisysTensor*)t)->tensor->data();
}
__export float* llaisysQwen2ModelGetLogits(struct LlaisysQwen2Model * model) {
    if (!model) return nullptr;
    return (float*)((Qwen2Impl*)model)->buf_logits->data();
}
}
