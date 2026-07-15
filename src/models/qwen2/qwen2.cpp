#include "qwen2.hpp"
#include "llaisys/ops.h"
#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/sample/op.hpp"
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

    const size_t q_dim = _meta.nh * _meta.dh;
    const size_t k_dim = _meta.nkvh * _meta.dh;
    const float scale = 1.0f / std::sqrt(static_cast<float>(_meta.dh));
    (void)scale;

    for(size_t i=0; i<_meta.nlayer; ++i) {
        _kv_caches.push_back({
            Tensor::create({_meta.maxseq, _meta.nkvh, _meta.dh}, _meta.dtype, _device_type, _device_id),
            Tensor::create({_meta.maxseq, _meta.nkvh, _meta.dh}, _meta.dtype, _device_type, _device_id)
        });

        auto q_flat = Tensor::create({1, q_dim}, _meta.dtype, _device_type, _device_id);
        auto k_flat = Tensor::create({1, k_dim}, _meta.dtype, _device_type, _device_id);
        auto v_flat = Tensor::create({1, k_dim}, _meta.dtype, _device_type, _device_id);
        auto attn_val_3d = Tensor::create({1, _meta.nh, _meta.dh}, _meta.dtype, _device_type, _device_id);

        _decode_layers.push_back({
            q_flat,
            k_flat,
            v_flat,
            q_flat->view({1, _meta.nh, _meta.dh}),
            k_flat->view({1, _meta.nkvh, _meta.dh}),
            v_flat->view({1, _meta.nkvh, _meta.dh}),
            attn_val_3d,
            attn_val_3d->view({1, _meta.hs}),
            Tensor::create({1, _meta.di}, _meta.dtype, _device_type, _device_id),
            Tensor::create({1, _meta.di}, _meta.dtype, _device_type, _device_id),
            Tensor::create({1, _meta.di}, _meta.dtype, _device_type, _device_id)
        });
    }

    _hidden_states = Tensor::create({1, _meta.hs}, _meta.dtype, _device_type, _device_id);
    _ln_out = Tensor::create({1, _meta.hs}, _meta.dtype, _device_type, _device_id);
    _attn_out = Tensor::create({1, _meta.hs}, _meta.dtype, _device_type, _device_id);
    _mlp_out = Tensor::create({1, _meta.hs}, _meta.dtype, _device_type, _device_id);
    _logits = Tensor::create({1, _meta.voc}, _meta.dtype, _device_type, _device_id);
    _tokens_tensor = Tensor::create({1}, LLAISYS_DTYPE_I64, _device_type, _device_id);
    _single_pos_id = Tensor::create({1}, LLAISYS_DTYPE_I64, _device_type, _device_id);
    _pos_ids = Tensor::create({_meta.maxseq}, LLAISYS_DTYPE_I64, _device_type, _device_id);
    _argmax_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, _device_type, _device_id);
    _argmax_val = Tensor::create({1}, _meta.dtype, _device_type, _device_id);
}

void Qwen2Model::ensure_prefill_buffers(size_t ntoken) {
    if (_prefill.capacity >= ntoken) {
        return;
    }

    const size_t next_capacity = std::max(ntoken, _prefill.capacity == 0 ? ntoken : _prefill.capacity * 2);
    const size_t q_dim = _meta.nh * _meta.dh;
    const size_t k_dim = _meta.nkvh * _meta.dh;

    _prefill.capacity = next_capacity;
    _prefill.tokens = Tensor::create({next_capacity}, LLAISYS_DTYPE_I64, _device_type, _device_id);
    _prefill.hidden_states = Tensor::create({next_capacity, _meta.hs}, _meta.dtype, _device_type, _device_id);
    _prefill.normed = Tensor::create({next_capacity, _meta.hs}, _meta.dtype, _device_type, _device_id);
    _prefill.q_flat = Tensor::create({next_capacity, q_dim}, _meta.dtype, _device_type, _device_id);
    _prefill.k_flat = Tensor::create({next_capacity, k_dim}, _meta.dtype, _device_type, _device_id);
    _prefill.v_flat = Tensor::create({next_capacity, k_dim}, _meta.dtype, _device_type, _device_id);
    _prefill.attn_val_3d = Tensor::create({next_capacity, _meta.nh, _meta.dh}, _meta.dtype, _device_type, _device_id);
    _prefill.attn_out = Tensor::create({next_capacity, _meta.hs}, _meta.dtype, _device_type, _device_id);
    _prefill.gate = Tensor::create({next_capacity, _meta.di}, _meta.dtype, _device_type, _device_id);
    _prefill.up = Tensor::create({next_capacity, _meta.di}, _meta.dtype, _device_type, _device_id);
    _prefill.swiglu = Tensor::create({next_capacity, _meta.di}, _meta.dtype, _device_type, _device_id);
    _prefill.mlp_out = Tensor::create({next_capacity, _meta.hs}, _meta.dtype, _device_type, _device_id);
}

tensor_t Qwen2Model::forward(const int64_t* token_ids, size_t ntoken) {
    CHECK_ARGUMENT(token_ids != nullptr, "token_ids must not be null");
    CHECK_ARGUMENT(ntoken > 0, "ntoken must be greater than zero");
    CHECK_ARGUMENT(_cur_pos + ntoken <= _meta.maxseq, "sequence exceeds KV-cache capacity");

    core::context().setDevice(_device_type, _device_id);

    if (ntoken == 1) {
        return forward_single_token(token_ids);
    }

    ensure_prefill_buffers(ntoken);

    tensor_t input_tokens = _prefill.tokens->slice(0, 0, ntoken);
    input_tokens->load(token_ids); 

    tensor_t current_pos_ids = _pos_ids->slice(0, 0, ntoken); 
    std::vector<int64_t> pos_data(ntoken);
    for(size_t i=0; i<ntoken; ++i) pos_data[i] = _cur_pos + i;
    current_pos_ids->load(pos_data.data()); 

    std::vector<size_t> seq_shape = {ntoken, _meta.hs};

    tensor_t hidden_states = _prefill.hidden_states->slice(0, 0, ntoken);
    hidden_states = hidden_states->view(seq_shape);
    ops::embedding(hidden_states, input_tokens, to_cpp(_weights.in_embed));
    
    const size_t q_dim = _meta.nh * _meta.dh;
    const size_t k_dim = _meta.nkvh * _meta.dh;
    const float scale = 1.0f / std::sqrt(static_cast<float>(_meta.dh));

    for(size_t i=0; i<_meta.nlayer; ++i) {
        tensor_t normed = _prefill.normed->slice(0, 0, ntoken);
        normed = normed->view(seq_shape);
        ops::rms_norm(normed, hidden_states, to_cpp(_weights.attn_norm_w, i), _meta.epsilon);

        tensor_t q = _prefill.q_flat->slice(0, 0, ntoken);
        q = q->view({ntoken, q_dim});
        tensor_t k = _prefill.k_flat->slice(0, 0, ntoken);
        k = k->view({ntoken, k_dim});
        tensor_t v = _prefill.v_flat->slice(0, 0, ntoken);
        v = v->view({ntoken, k_dim});

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

        tensor_t attn_val = _prefill.attn_val_3d->slice(0, 0, ntoken);
        attn_val = attn_val->view({ntoken, _meta.nh, _meta.dh});
        
        ops::self_attention(attn_val, q, k_full, v_full, scale);

        attn_val = attn_val->view({ntoken, _meta.hs});
        
        tensor_t attn_output = _prefill.attn_out->slice(0, 0, ntoken);
        attn_output = attn_output->view(seq_shape);
        ops::linear(attn_output, attn_val, to_cpp(_weights.attn_o_w, i), nullptr); 

        ops::add(hidden_states, hidden_states, attn_output);

        normed = _prefill.normed->slice(0, 0, ntoken);
        normed = normed->view(seq_shape);
        ops::rms_norm(normed, hidden_states, to_cpp(_weights.mlp_norm_w, i), _meta.epsilon);

        tensor_t gate = _prefill.gate->slice(0, 0, ntoken);
        gate = gate->view({ntoken, _meta.di});
        tensor_t up = _prefill.up->slice(0, 0, ntoken);
        up = up->view({ntoken, _meta.di});
        
        ops::linear(gate, normed, to_cpp(_weights.mlp_gate_w, i), nullptr);
        ops::linear(up, normed, to_cpp(_weights.mlp_up_w, i), nullptr);

        tensor_t swiglu_out = _prefill.swiglu->slice(0, 0, ntoken);
        swiglu_out = swiglu_out->view({ntoken, _meta.di});
        ops::swiglu(swiglu_out, gate, up);

        tensor_t mlp_output = _prefill.mlp_out->slice(0, 0, ntoken);
        mlp_output = mlp_output->view(seq_shape);
        ops::linear(mlp_output, swiglu_out, to_cpp(_weights.mlp_down_w, i), nullptr);

        ops::add(hidden_states, hidden_states, mlp_output);
    }

    tensor_t final_normed = _prefill.normed->slice(0, 0, ntoken);
    final_normed = final_normed->view(seq_shape);
    ops::rms_norm(final_normed, hidden_states, to_cpp(_weights.out_norm_w), _meta.epsilon);

    tensor_t last_hidden = final_normed->slice(0, ntoken-1, ntoken); 
    last_hidden = last_hidden->view({1, _meta.hs});

    ops::linear(_logits, last_hidden, to_cpp(_weights.out_embed), nullptr);

    _cur_pos += ntoken;

    return _logits;
}

tensor_t Qwen2Model::forward_single_token(const int64_t* token_id) {
    CHECK_ARGUMENT(token_id != nullptr, "token_id must not be null");

    _tokens_tensor->load(token_id);
    const int64_t current_pos = static_cast<int64_t>(_cur_pos);
    _single_pos_id->load(&current_pos);

    ops::embedding(_hidden_states, _tokens_tensor, to_cpp(_weights.in_embed));

    const float scale = 1.0f / std::sqrt(static_cast<float>(_meta.dh));
    for (size_t i = 0; i < _meta.nlayer; ++i) {
        auto &layer = _decode_layers[i];

        ops::rms_norm(_ln_out, _hidden_states, to_cpp(_weights.attn_norm_w, i), _meta.epsilon);

        ops::linear(layer.q_flat, _ln_out, to_cpp(_weights.attn_q_w, i), to_cpp(_weights.attn_q_b, i));
        ops::linear(layer.k_flat, _ln_out, to_cpp(_weights.attn_k_w, i), to_cpp(_weights.attn_k_b, i));
        ops::linear(layer.v_flat, _ln_out, to_cpp(_weights.attn_v_w, i), to_cpp(_weights.attn_v_b, i));

        ops::rope(layer.q_view, layer.q_view, _single_pos_id, _meta.theta);
        ops::rope(layer.k_view, layer.k_view, _single_pos_id, _meta.theta);

        tensor_t k_cache_slot = _kv_caches[i].k->slice(0, _cur_pos, _cur_pos + 1);
        tensor_t v_cache_slot = _kv_caches[i].v->slice(0, _cur_pos, _cur_pos + 1);
        ops::rearrange(k_cache_slot, layer.k_view);
        ops::rearrange(v_cache_slot, layer.v_view);

        tensor_t k_full = _kv_caches[i].k->slice(0, 0, _cur_pos + 1);
        tensor_t v_full = _kv_caches[i].v->slice(0, 0, _cur_pos + 1);
        ops::self_attention(layer.attn_val_3d, layer.q_view, k_full, v_full, scale);

        ops::linear(_attn_out, layer.attn_val_2d, to_cpp(_weights.attn_o_w, i), nullptr);
        ops::add(_hidden_states, _hidden_states, _attn_out);

        ops::rms_norm(_ln_out, _hidden_states, to_cpp(_weights.mlp_norm_w, i), _meta.epsilon);
        ops::linear(layer.gate, _ln_out, to_cpp(_weights.mlp_gate_w, i), nullptr);
        ops::linear(layer.up, _ln_out, to_cpp(_weights.mlp_up_w, i), nullptr);
        ops::swiglu(layer.swiglu, layer.gate, layer.up);
        ops::linear(_mlp_out, layer.swiglu, to_cpp(_weights.mlp_down_w, i), nullptr);
        ops::add(_hidden_states, _hidden_states, _mlp_out);
    }

    ops::rms_norm(_ln_out, _hidden_states, to_cpp(_weights.out_norm_w), _meta.epsilon);
    ops::linear(_logits, _ln_out, to_cpp(_weights.out_embed), nullptr);

    _cur_pos += 1;
    return _logits;
}

int64_t Qwen2Model::infer(const int64_t* token_ids, size_t ntoken) {
    tensor_t logits = forward(token_ids, ntoken);
    ops::argmax(_argmax_idx, _argmax_val, logits);

    if (_device_type == LLAISYS_DEVICE_CPU) {
        return *reinterpret_cast<int64_t*>(_argmax_idx->data());
    }
    int64_t result = 0;
    core::context().runtime().api()->memcpy_sync(
        &result,
        _argmax_idx->data(),
        sizeof(result),
        LLAISYS_MEMCPY_D2H);
    return result;
}

int64_t Qwen2Model::generateNext(const int64_t* token_ids, size_t ntoken, int top_k, float top_p, float temperature) {
    tensor_t logits = forward(token_ids, ntoken);
    return ops::sample(logits, top_k, top_p, temperature);
}

void Qwen2Model::reset() {
    _cur_pos = 0;
}

void Qwen2Model::truncate(size_t position) {
    CHECK_ARGUMENT(position <= _cur_pos, "truncate position exceeds current cache length");
    _cur_pos = position;
}

} // namespace llaisys
