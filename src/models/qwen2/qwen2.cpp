#include "qwen2.hpp"
#include "../../core/context/context.hpp" // 新增：用于访问 memcpy_sync
#include <iostream>
#include <cmath>
#include <cstring> // for memcpy if needed, but we use runtime api

namespace llaisys::models::qwen2 {

Qwen2Model::Qwen2Model(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int device_id) {
    _meta = *meta;
    init_weights(device, device_id);
    init_kv_cache(device, device_id);
}

void Qwen2Model::init_weights(llaisysDeviceType_t device, int device_id) {
    auto dtype = _meta.dtype;
    
    // Helper to create empty tensor
    auto mk = [&](const std::vector<size_t>& shape) {
        return Tensor::create(shape, dtype, device, device_id);
    };

    in_embed = mk({_meta.voc, _meta.hs});
    out_embed = mk({_meta.voc, _meta.hs});
    out_norm_w = mk({_meta.hs});

    auto init_layer_w = [&](std::vector<tensor_t>& vec, const std::vector<size_t>& shape) {
        vec.resize(_meta.nlayer);
        for(size_t i=0; i<_meta.nlayer; ++i) vec[i] = mk(shape);
    };

    init_layer_w(attn_norm_w, {_meta.hs});
    init_layer_w(attn_q_w, {_meta.nh * _meta.dh, _meta.hs});
    init_layer_w(attn_q_b, {_meta.nh * _meta.dh});
    init_layer_w(attn_k_w, {_meta.nkvh * _meta.dh, _meta.hs});
    init_layer_w(attn_k_b, {_meta.nkvh * _meta.dh});
    init_layer_w(attn_v_w, {_meta.nkvh * _meta.dh, _meta.hs});
    init_layer_w(attn_v_b, {_meta.nkvh * _meta.dh});
    init_layer_w(attn_o_w, {_meta.hs, _meta.nh * _meta.dh});
    
    init_layer_w(mlp_norm_w, {_meta.hs});
    init_layer_w(mlp_gate_w, {_meta.di, _meta.hs});
    init_layer_w(mlp_up_w, {_meta.di, _meta.hs});
    init_layer_w(mlp_down_w, {_meta.hs, _meta.di});
}

void Qwen2Model::init_kv_cache(llaisysDeviceType_t device, int device_id) {
    k_cache.resize(_meta.nlayer);
    v_cache.resize(_meta.nlayer);
    for(size_t i=0; i<_meta.nlayer; ++i) {
        // [max_seq, nkvh, dh]
        k_cache[i] = Tensor::create({_meta.maxseq, _meta.nkvh, _meta.dh}, _meta.dtype, device, device_id);
        v_cache[i] = Tensor::create({_meta.maxseq, _meta.nkvh, _meta.dh}, _meta.dtype, device, device_id);
    }
}

int64_t Qwen2Model::forward(const int64_t *token_ids_ptr, size_t ntoken) {
    auto device = in_embed->deviceType();
    int device_id = in_embed->deviceId();
    auto dtype = _meta.dtype;

    // 0. Inputs
    auto input_ids_host = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
    input_ids_host->load(token_ids_ptr);
    
    tensor_t input_ids;
    if (device == LLAISYS_DEVICE_CPU) {
        input_ids = input_ids_host;
    } else {
        input_ids = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, device, device_id);
        llaisys::core::context().runtime().api()->memcpy_sync(input_ids->data(), input_ids_host->data(), ntoken * sizeof(int64_t), LLAISYS_MEMCPY_H2D);
    }

    std::vector<int64_t> pos_vec(ntoken);
    for(size_t i=0; i<ntoken; ++i) pos_vec[i] = _current_pos + i;
    auto pos_ids_host = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
    pos_ids_host->load(pos_vec.data());
    
    tensor_t pos_ids;
    if (device == LLAISYS_DEVICE_CPU) {
        pos_ids = pos_ids_host;
    } else {
        pos_ids = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, device, device_id);
        llaisys::core::context().runtime().api()->memcpy_sync(pos_ids->data(), pos_ids_host->data(), ntoken * sizeof(int64_t), LLAISYS_MEMCPY_H2D);
    }

    auto mk = [&](const std::vector<size_t>& s) { return Tensor::create(s, dtype, device, device_id); };

    // 1. Embedding
    auto hidden_states = mk({ntoken, _meta.hs});
    ops::embedding(hidden_states, input_ids, in_embed);

    // 2. Layers
    for (size_t i = 0; i < _meta.nlayer; ++i) {
        auto residual = hidden_states; 
        
        // Attention Block
        auto hidden_norm = mk({ntoken, _meta.hs});
        ops::rms_norm(hidden_norm, hidden_states, attn_norm_w[i], _meta.epsilon);

        auto q_flat = mk({ntoken, _meta.nh * _meta.dh});
        ops::linear(q_flat, hidden_norm, attn_q_w[i], attn_q_b[i]);
        auto q = q_flat->view({ntoken, _meta.nh, _meta.dh});

        auto k_flat = mk({ntoken, _meta.nkvh * _meta.dh});
        ops::linear(k_flat, hidden_norm, attn_k_w[i], attn_k_b[i]);
        auto k = k_flat->view({ntoken, _meta.nkvh, _meta.dh});

        auto v_flat = mk({ntoken, _meta.nkvh * _meta.dh});
        ops::linear(v_flat, hidden_norm, attn_v_w[i], attn_v_b[i]);
        auto v = v_flat->view({ntoken, _meta.nkvh, _meta.dh});

        ops::rope(q, q, pos_ids, _meta.theta);
        ops::rope(k, k, pos_ids, _meta.theta);

        // KV Cache Update
        auto k_slot = k_cache[i]->slice(0, _current_pos, _current_pos + ntoken);
        auto v_slot = v_cache[i]->slice(0, _current_pos, _current_pos + ntoken);
        
        llaisysMemcpyKind_t kind = (device == LLAISYS_DEVICE_CPU) ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_D2D;
        size_t copy_bytes = k->numel() * k->elementSize();
        
        // 执行拷贝: k -> k_slot
        llaisys::core::context().runtime().api()->memcpy_sync(
            k_slot->data(), 
            k->data(), 
            copy_bytes, 
            kind
        );
        // 执行拷贝: v -> v_slot
        llaisys::core::context().runtime().api()->memcpy_sync(
            v_slot->data(), 
            v->data(), 
            copy_bytes, 
            kind
        );

        // Attention
        auto k_active = k_cache[i]->slice(0, 0, _current_pos + ntoken);
        auto v_active = v_cache[i]->slice(0, 0, _current_pos + ntoken);

        auto attn_out_view = mk({ntoken, _meta.nh, _meta.dh});
        float scale = 1.0f / std::sqrt(static_cast<float>(_meta.dh));
        ops::self_attention(attn_out_view, q, k_active, v_active, scale);

        auto attn_out_flat = attn_out_view->view({ntoken, _meta.nh * _meta.dh});
        auto attn_proj = mk({ntoken, _meta.hs});
        ops::linear(attn_proj, attn_out_flat, attn_o_w[i], nullptr); 

        // Residual Add
        ops::add(hidden_states, hidden_states, attn_proj);

        // MLP Block
        auto mlp_norm = mk({ntoken, _meta.hs});
        ops::rms_norm(mlp_norm, hidden_states, mlp_norm_w[i], _meta.epsilon);

        auto gate = mk({ntoken, _meta.di});
        ops::linear(gate, mlp_norm, mlp_gate_w[i], nullptr);
        
        auto up = mk({ntoken, _meta.di});
        ops::linear(up, mlp_norm, mlp_up_w[i], nullptr);

        auto mlp_act = mk({ntoken, _meta.di});
        ops::swiglu(mlp_act, gate, up);

        auto mlp_out = mk({ntoken, _meta.hs});
        ops::linear(mlp_out, mlp_act, mlp_down_w[i], nullptr);

        ops::add(hidden_states, hidden_states, mlp_out);
    }

    // 3. Final
    auto last_hidden = hidden_states->slice(0, ntoken - 1, ntoken);
    auto final_norm = mk({1, _meta.hs});
    ops::rms_norm(final_norm, last_hidden, out_norm_w, _meta.epsilon);

    auto logits = mk({1, _meta.voc});
    ops::linear(logits, final_norm, out_embed, nullptr);

    auto max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, device, device_id);
    auto max_val = mk({1});
    ops::argmax(max_idx, max_val, logits);

    int64_t next_token;
    llaisys::core::context().runtime().api()->memcpy_sync(&next_token, max_idx->data(), sizeof(int64_t), LLAISYS_MEMCPY_D2H);

    _current_pos += ntoken;
    return next_token;
}

}