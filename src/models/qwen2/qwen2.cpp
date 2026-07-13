#include "qwen2.hpp"
#include "../../llaisys/llaisys_tensor.hpp" // 用于 LlaisysTensor 包装器定义
#include "../../utils.hpp"
#include <cstring>
#include <cmath>
#include <iostream>
#include <vector>

// 引入算子
#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"

namespace llaisys::models::qwen2 {

// 辅助函数：将 C++ tensor_t 包装为 C API 的 llaisysTensor_t
llaisysTensor_t wrap(tensor_t t) {
    return new LlaisysTensor{t};
}

Qwen2Model::Qwen2Model(const LlaisysQwen2Meta &meta, llaisysDeviceType_t device_type, int device_id)
    : _meta(meta), _device_type(device_type), _device_id(device_id), _current_pos(0) {
    
    // 设置上下文设备
    core::context().setDevice(device_type, device_id);

    // 1. 初始化基础权重
    _in_embed = create_weight({meta.voc, meta.hs});
    _out_embed = create_weight({meta.voc, meta.hs});
    _out_norm_w = create_weight({meta.hs});

    _weights_export.in_embed = wrap(_in_embed);
    _weights_export.out_embed = wrap(_out_embed);
    _weights_export.out_norm_w = wrap(_out_norm_w);

    // 2. 分配层级权重数组
    _weights_export.attn_norm_w = new llaisysTensor_t[meta.nlayer];
    _weights_export.attn_q_w = new llaisysTensor_t[meta.nlayer];
    _weights_export.attn_q_b = new llaisysTensor_t[meta.nlayer];
    _weights_export.attn_k_w = new llaisysTensor_t[meta.nlayer];
    _weights_export.attn_k_b = new llaisysTensor_t[meta.nlayer];
    _weights_export.attn_v_w = new llaisysTensor_t[meta.nlayer];
    _weights_export.attn_v_b = new llaisysTensor_t[meta.nlayer];
    _weights_export.attn_o_w = new llaisysTensor_t[meta.nlayer];
    _weights_export.mlp_norm_w = new llaisysTensor_t[meta.nlayer];
    _weights_export.mlp_gate_w = new llaisysTensor_t[meta.nlayer];
    _weights_export.mlp_up_w = new llaisysTensor_t[meta.nlayer];
    _weights_export.mlp_down_w = new llaisysTensor_t[meta.nlayer];

    // 修复 unused variable 'head_dim' 错误：
    // 直接在下方使用 meta.dh，不定义局部变量 head_dim

    for (size_t i = 0; i < meta.nlayer; ++i) {
        // --- Allocation ---
        auto attn_norm = create_weight({meta.hs});
        auto mlp_norm = create_weight({meta.hs});

        // 使用 meta.dh 替代 head_dim
        auto q_w = create_weight({meta.nh * meta.dh, meta.hs});
        auto q_b = create_weight({meta.nh * meta.dh});
        auto k_w = create_weight({meta.nkvh * meta.dh, meta.hs});
        auto k_b = create_weight({meta.nkvh * meta.dh});
        auto v_w = create_weight({meta.nkvh * meta.dh, meta.hs});
        auto v_b = create_weight({meta.nkvh * meta.dh});
        auto o_w = create_weight({meta.hs, meta.nh * meta.dh}); 

        auto g_w = create_weight({meta.di, meta.hs});
        auto u_w = create_weight({meta.di, meta.hs});
        auto d_w = create_weight({meta.hs, meta.di});

        // --- Store internal shared_ptrs ---
        _layers_input_norm.push_back(attn_norm);
        _layers_post_norm.push_back(mlp_norm);
        _layers_q_w.push_back(q_w); _layers_q_b.push_back(q_b);
        _layers_k_w.push_back(k_w); _layers_k_b.push_back(k_b);
        _layers_v_w.push_back(v_w); _layers_v_b.push_back(v_b);
        _layers_o_w.push_back(o_w);
        _layers_gate_w.push_back(g_w);
        _layers_up_w.push_back(u_w);
        _layers_down_w.push_back(d_w);

        // --- Export wrappers ---
        _weights_export.attn_norm_w[i] = wrap(attn_norm);
        _weights_export.mlp_norm_w[i] = wrap(mlp_norm);
        _weights_export.attn_q_w[i] = wrap(q_w);
        _weights_export.attn_q_b[i] = wrap(q_b);
        _weights_export.attn_k_w[i] = wrap(k_w);
        _weights_export.attn_k_b[i] = wrap(k_b);
        _weights_export.attn_v_w[i] = wrap(v_w);
        _weights_export.attn_v_b[i] = wrap(v_b);
        _weights_export.attn_o_w[i] = wrap(o_w);
        _weights_export.mlp_gate_w[i] = wrap(g_w);
        _weights_export.mlp_up_w[i] = wrap(u_w);
        _weights_export.mlp_down_w[i] = wrap(d_w);

        // --- KV Cache ---
        // 使用 meta.dh
        auto k_c = Tensor::create({meta.maxseq, meta.nkvh, meta.dh}, meta.dtype, device_type, device_id);
        auto v_c = Tensor::create({meta.maxseq, meta.nkvh, meta.dh}, meta.dtype, device_type, device_id);
        
        _k_cache.push_back(k_c);
        _v_cache.push_back(v_c);
    }
}

Qwen2Model::~Qwen2Model() {
    auto free_arr = [](llaisysTensor_t *arr, size_t n) {
        for(size_t i=0; i<n; i++) delete arr[i]; 
        delete[] arr;
    };

    delete _weights_export.in_embed;
    delete _weights_export.out_embed;
    delete _weights_export.out_norm_w;

    free_arr(_weights_export.attn_norm_w, _meta.nlayer);
    free_arr(_weights_export.attn_q_w, _meta.nlayer);
    free_arr(_weights_export.attn_q_b, _meta.nlayer);
    free_arr(_weights_export.attn_k_w, _meta.nlayer);
    free_arr(_weights_export.attn_k_b, _meta.nlayer);
    free_arr(_weights_export.attn_v_w, _meta.nlayer);
    free_arr(_weights_export.attn_v_b, _meta.nlayer);
    free_arr(_weights_export.attn_o_w, _meta.nlayer);
    free_arr(_weights_export.mlp_norm_w, _meta.nlayer);
    free_arr(_weights_export.mlp_gate_w, _meta.nlayer);
    free_arr(_weights_export.mlp_up_w, _meta.nlayer);
    free_arr(_weights_export.mlp_down_w, _meta.nlayer);
}

tensor_t Qwen2Model::create_weight(const std::vector<size_t>& shape) {
    return Tensor::create(shape, _meta.dtype, _device_type, _device_id);
}

LlaisysQwen2Weights *Qwen2Model::weights() {
    return &_weights_export;
}

int64_t Qwen2Model::infer(int64_t *token_ids, size_t ntoken) {
    core::context().setDevice(_device_type, _device_id);
    auto &runtime = core::context().runtime();

    // 1. Inputs [ntoken]
    auto input_tokens = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, _device_type, _device_id);
    input_tokens->load(token_ids); 

    // 生成 Position IDs
    auto pos_ids = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, _device_type, _device_id);
    std::vector<int64_t> pos_data(ntoken);
    for(size_t i=0; i<ntoken; ++i) pos_data[i] = _current_pos + i;
    pos_ids->load(pos_data.data());

    // 2. Embedding [ntoken, hs]
    auto x = Tensor::create({ntoken, _meta.hs}, _meta.dtype, _device_type, _device_id);
    ops::embedding(x, input_tokens, _in_embed);

    // 3. Transformer Layers
    for (size_t i = 0; i < _meta.nlayer; ++i) {
        auto residual = x; 

        // --- Attention Block ---
        // Norm
        auto x_norm = Tensor::create({ntoken, _meta.hs}, _meta.dtype, _device_type, _device_id);
        ops::rms_norm(x_norm, x, _layers_input_norm[i], _meta.epsilon);

        // QKV Projection
        auto q_flat = Tensor::create({ntoken, _meta.nh * _meta.dh}, _meta.dtype, _device_type, _device_id);
        auto k_flat = Tensor::create({ntoken, _meta.nkvh * _meta.dh}, _meta.dtype, _device_type, _device_id);
        auto v_flat = Tensor::create({ntoken, _meta.nkvh * _meta.dh}, _meta.dtype, _device_type, _device_id);

        ops::linear(q_flat, x_norm, _layers_q_w[i], _layers_q_b[i]);
        ops::linear(k_flat, x_norm, _layers_k_w[i], _layers_k_b[i]);
        ops::linear(v_flat, x_norm, _layers_v_w[i], _layers_v_b[i]);

        // Reshape & RoPE
        auto q = q_flat->view({ntoken, _meta.nh, _meta.dh});
        auto k = k_flat->view({ntoken, _meta.nkvh, _meta.dh});
        auto v = v_flat->view({ntoken, _meta.nkvh, _meta.dh});

        ops::rope(q, q, pos_ids, _meta.theta);
        ops::rope(k, k, pos_ids, _meta.theta);

        // KV Cache Update
        auto k_cache_slot = _k_cache[i]->slice(0, _current_pos, _current_pos + ntoken);
        auto v_cache_slot = _v_cache[i]->slice(0, _current_pos, _current_pos + ntoken);
        
        runtime.api()->memcpy_sync(k_cache_slot->data(), k->data(), k->numel() * k->elementSize(), LLAISYS_MEMCPY_D2D);
        runtime.api()->memcpy_sync(v_cache_slot->data(), v->data(), v->numel() * v->elementSize(), LLAISYS_MEMCPY_D2D);

        // Attention
        auto k_full = _k_cache[i]->slice(0, 0, _current_pos + ntoken);
        auto v_full = _v_cache[i]->slice(0, 0, _current_pos + ntoken);

        auto attn_out = Tensor::create({ntoken, _meta.nh, _meta.dh}, _meta.dtype, _device_type, _device_id);
        // std::sqrt 需要 <cmath>
        float scale = 1.0f / std::sqrt(static_cast<float>(_meta.dh));
        
        ops::self_attention(attn_out, q, k_full, v_full, scale);

        // Output Projection
        auto attn_out_flat = attn_out->view({ntoken, _meta.nh * _meta.dh});
        auto h_attn = Tensor::create({ntoken, _meta.hs}, _meta.dtype, _device_type, _device_id);
        ops::linear(h_attn, attn_out_flat, _layers_o_w[i], nullptr);

        // Residual Add
        ops::add(x, residual, h_attn);
        residual = x; 

        // --- MLP Block ---
        // Norm
        ops::rms_norm(x_norm, x, _layers_post_norm[i], _meta.epsilon);

        // Gate & Up
        auto gate = Tensor::create({ntoken, _meta.di}, _meta.dtype, _device_type, _device_id);
        auto up = Tensor::create({ntoken, _meta.di}, _meta.dtype, _device_type, _device_id);
        ops::linear(gate, x_norm, _layers_gate_w[i], nullptr);
        ops::linear(up, x_norm, _layers_up_w[i], nullptr);

        // SwiGLU
        auto mlp_act = Tensor::create({ntoken, _meta.di}, _meta.dtype, _device_type, _device_id);
        ops::swiglu(mlp_act, gate, up);

        // Down
        auto h_mlp = Tensor::create({ntoken, _meta.hs}, _meta.dtype, _device_type, _device_id);
        ops::linear(h_mlp, mlp_act, _layers_down_w[i], nullptr);

        // Residual Add
        ops::add(x, residual, h_mlp);
    }

    // 4. Final Norm
    auto x_final = Tensor::create({ntoken, _meta.hs}, _meta.dtype, _device_type, _device_id);
    ops::rms_norm(x_final, x, _out_norm_w, _meta.epsilon);

    // 5. LM Head
    auto x_last = x_final->slice(0, ntoken - 1, ntoken); 
    auto logits = Tensor::create({1, _meta.voc}, _meta.dtype, _device_type, _device_id);
    ops::linear(logits, x_last, _out_embed, nullptr); 

    // 6. Argmax
    auto max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, _device_type, _device_id);
    auto max_val = Tensor::create({1}, _meta.dtype, _device_type, _device_id); 
    ops::argmax(max_idx, max_val, logits->view({_meta.voc}));

    int64_t next_token = 0;
    runtime.api()->memcpy_sync(&next_token, max_idx->data(), sizeof(int64_t), LLAISYS_MEMCPY_D2H);

    _current_pos += ntoken;

    return next_token;
}

} // namespace