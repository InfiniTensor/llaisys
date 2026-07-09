#include "qwen2.hpp"
#include "llaisys/ops.h"
#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rearrange/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils/check.hpp"
#include <iostream>
#include <cstdlib>
#include <cmath> 
#include "../../llaisys/llaisys_tensor.hpp"

namespace llaisys {

inline tensor_t to_cpp(llaisysTensor_t t) {
    if (!t) return nullptr;
    return reinterpret_cast<LlaisysTensor*>(t)->tensor;
}

inline tensor_t to_cpp(llaisysTensor_t* t_array, size_t idx) {
    if (!t_array || !t_array[idx]) return nullptr;
    return reinterpret_cast<LlaisysTensor*>(t_array[idx])->tensor;
}

Qwen2Model::Qwen2Model(const LlaisysQwen2Meta& meta, llaisysDeviceType_t device, int device_id)
    : _meta(meta), _device_type(device), _device_id(device_id) {
    
    _weights.attn_norm_w = (llaisysTensor_t*)calloc(meta.nlayer, sizeof(llaisysTensor_t));
    _weights.attn_q_w = (llaisysTensor_t*)calloc(meta.nlayer, sizeof(llaisysTensor_t));
    _weights.attn_q_b = (llaisysTensor_t*)calloc(meta.nlayer, sizeof(llaisysTensor_t));
    _weights.attn_k_w = (llaisysTensor_t*)calloc(meta.nlayer, sizeof(llaisysTensor_t));
    _weights.attn_k_b = (llaisysTensor_t*)calloc(meta.nlayer, sizeof(llaisysTensor_t));
    _weights.attn_v_w = (llaisysTensor_t*)calloc(meta.nlayer, sizeof(llaisysTensor_t));
    _weights.attn_v_b = (llaisysTensor_t*)calloc(meta.nlayer, sizeof(llaisysTensor_t));
    _weights.attn_o_w = (llaisysTensor_t*)calloc(meta.nlayer, sizeof(llaisysTensor_t));
    _weights.mlp_norm_w = (llaisysTensor_t*)calloc(meta.nlayer, sizeof(llaisysTensor_t));
    _weights.mlp_gate_w = (llaisysTensor_t*)calloc(meta.nlayer, sizeof(llaisysTensor_t));
    _weights.mlp_up_w = (llaisysTensor_t*)calloc(meta.nlayer, sizeof(llaisysTensor_t));
    _weights.mlp_down_w = (llaisysTensor_t*)calloc(meta.nlayer, sizeof(llaisysTensor_t));

    init_buffers();
}

Qwen2Model::~Qwen2Model() {
    free(_weights.attn_norm_w);
    free(_weights.attn_q_w);
    free(_weights.attn_q_b);
    free(_weights.attn_k_w);
    free(_weights.attn_k_b);
    free(_weights.attn_v_w);
    free(_weights.attn_v_b);
    free(_weights.attn_o_w);
    free(_weights.mlp_norm_w);
    free(_weights.mlp_gate_w);
    free(_weights.mlp_up_w);
    free(_weights.mlp_down_w);
}

void Qwen2Model::init_buffers() {
    core::context().setDevice(_device_type, _device_id);

    for(size_t i=0; i<_meta.nlayer; ++i) {
        _kv_caches.push_back({
            Tensor::create({_meta.maxseq, _meta.nkvh, _meta.dh}, _meta.dtype, _device_type, _device_id),
            Tensor::create({_meta.maxseq, _meta.nkvh, _meta.dh}, _meta.dtype, _device_type, _device_id)
        });
    }

    _hidden_states = Tensor::create({1, 1, _meta.hs}, _meta.dtype, _device_type, _device_id);
    _residual = Tensor::create({1, 1, _meta.hs}, _meta.dtype, _device_type, _device_id);
    _ln_out = Tensor::create({1, 1, _meta.hs}, _meta.dtype, _device_type, _device_id);
    _attn_out = Tensor::create({1, 1, _meta.hs}, _meta.dtype, _device_type, _device_id);
    _mlp_out = Tensor::create({1, 1, _meta.hs}, _meta.dtype, _device_type, _device_id);
    
    _logits = Tensor::create({1, _meta.voc}, _meta.dtype, _device_type, _device_id);
    
    _pos_ids = Tensor::create({_meta.maxseq}, LLAISYS_DTYPE_I64, _device_type, _device_id);
}

int64_t Qwen2Model::infer(const int64_t* token_ids, size_t ntoken) {
    core::context().setDevice(_device_type, _device_id);
    
    tensor_t input_tokens = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, _device_type, _device_id);
    input_tokens->load(token_ids); 

    tensor_t current_pos_ids = _pos_ids->slice(0, 0, ntoken); 
    std::vector<int64_t> pos_data(ntoken);
    for(size_t i=0; i<ntoken; ++i) pos_data[i] = _cur_pos + i;
    current_pos_ids->load(pos_data.data()); 

    std::vector<size_t> seq_shape = {ntoken, _meta.hs};
    
    tensor_t hidden_states = Tensor::create({ntoken, _meta.hs}, _meta.dtype, _device_type, _device_id);
    ops::embedding(hidden_states, input_tokens, to_cpp(_weights.in_embed));
    
    for(size_t i=0; i<_meta.nlayer; ++i) {
        tensor_t normed = Tensor::create(seq_shape, _meta.dtype, _device_type, _device_id);
        ops::rms_norm(normed, hidden_states, to_cpp(_weights.attn_norm_w, i), _meta.epsilon);

        size_t q_dim = _meta.nh * _meta.dh;
        size_t k_dim = _meta.nkvh * _meta.dh;
        
        tensor_t q = Tensor::create({ntoken, q_dim}, _meta.dtype, _device_type, _device_id);
        tensor_t k = Tensor::create({ntoken, k_dim}, _meta.dtype, _device_type, _device_id);
        tensor_t v = Tensor::create({ntoken, k_dim}, _meta.dtype, _device_type, _device_id);

        ops::linear(q, normed, to_cpp(_weights.attn_q_w, i), to_cpp(_weights.attn_q_b, i));
        ops::linear(k, normed, to_cpp(_weights.attn_k_w, i), to_cpp(_weights.attn_k_b, i));
        ops::linear(v, normed, to_cpp(_weights.attn_v_w, i), to_cpp(_weights.attn_v_b, i));

        q = q->view({ntoken, _meta.nh, _meta.dh});
        k = k->view({ntoken, _meta.nkvh, _meta.dh});
        v = v->view({ntoken, _meta.nkvh, _meta.dh});

        ops::rope(q, q, current_pos_ids, _meta.theta);
        ops::rope(k, k, current_pos_ids, _meta.theta);

        tensor_t k_cache_slot = _kv_caches[i].k->slice(0, _cur_pos, _cur_pos + ntoken);
        tensor_t v_cache_slot = _kv_caches[i].v->slice(0, _cur_pos, _cur_pos + ntoken);
        
        ops::rearrange(k_cache_slot, k);
        ops::rearrange(v_cache_slot, v);

        tensor_t k_full = _kv_caches[i].k->slice(0, 0, _cur_pos + ntoken);
        tensor_t v_full = _kv_caches[i].v->slice(0, 0, _cur_pos + ntoken);

        tensor_t attn_val = Tensor::create({ntoken, _meta.nh, _meta.dh}, _meta.dtype, _device_type, _device_id);
        float scale = 1.0f / std::sqrt((float)_meta.dh);
        
        ops::self_attention(attn_val, q, k_full, v_full, scale);

        attn_val = attn_val->view({ntoken, _meta.hs});
        
        tensor_t attn_output = Tensor::create(seq_shape, _meta.dtype, _device_type, _device_id);
        ops::linear(attn_output, attn_val, to_cpp(_weights.attn_o_w, i), nullptr); 

        ops::add(hidden_states, hidden_states, attn_output);

        normed = Tensor::create(seq_shape, _meta.dtype, _device_type, _device_id);
        ops::rms_norm(normed, hidden_states, to_cpp(_weights.mlp_norm_w, i), _meta.epsilon);

        tensor_t gate = Tensor::create({ntoken, _meta.di}, _meta.dtype, _device_type, _device_id);
        tensor_t up = Tensor::create({ntoken, _meta.di}, _meta.dtype, _device_type, _device_id);
        
        ops::linear(gate, normed, to_cpp(_weights.mlp_gate_w, i), nullptr);
        ops::linear(up, normed, to_cpp(_weights.mlp_up_w, i), nullptr);

        tensor_t swiglu_out = Tensor::create({ntoken, _meta.di}, _meta.dtype, _device_type, _device_id);
        ops::swiglu(swiglu_out, gate, up);

        tensor_t mlp_output = Tensor::create(seq_shape, _meta.dtype, _device_type, _device_id);
        ops::linear(mlp_output, swiglu_out, to_cpp(_weights.mlp_down_w, i), nullptr);

        ops::add(hidden_states, hidden_states, mlp_output);
    }

    tensor_t final_normed = Tensor::create(seq_shape, _meta.dtype, _device_type, _device_id);
    ops::rms_norm(final_normed, hidden_states, to_cpp(_weights.out_norm_w), _meta.epsilon);

    tensor_t last_hidden = final_normed->slice(0, ntoken-1, ntoken); 
    last_hidden = last_hidden->view({1, _meta.hs});

    tensor_t logits = Tensor::create({1, _meta.voc}, _meta.dtype, _device_type, _device_id);
    ops::linear(logits, last_hidden, to_cpp(_weights.out_embed), nullptr); 

    tensor_t max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, _device_type, _device_id);
    tensor_t max_val = Tensor::create({1}, _meta.dtype, _device_type, _device_id); 
    
    ops::argmax(max_idx, max_val, logits);

    int64_t next_token_id;
    if (_device_type == LLAISYS_DEVICE_CPU) {
        next_token_id = *reinterpret_cast<int64_t*>(max_idx->data());
    } else {
        next_token_id = 0; 
    }

    _cur_pos += ntoken;

    return next_token_id;
}

} // namespace llaisys
