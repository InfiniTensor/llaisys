#include "qwen2.hpp"

#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rearrange/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"

#include <cmath>
#include <cstring>

namespace llaisys::models {

Qwen2Model::Qwen2Model(const Qwen2Meta* meta, llaisysDeviceType_t device, int device_id)
    : _meta(*meta), _device_type(device), _device_id(device_id), _cache_len(0), _max_batch(0) {

    // Set context to the correct device
    core::context().setDevice(device, device_id);

    size_t nlayer = _meta.nlayer;
    size_t hs = _meta.hs;
    size_t nh = _meta.nh;
    size_t nkvh = _meta.nkvh;
    size_t dh = _meta.dh;
    size_t di = _meta.di;
    size_t maxseq = _meta.maxseq;
    size_t voc = _meta.voc;

    // Allocate embedding weights
    _weights.in_embed = Tensor::create({voc, hs}, LLAISYS_DTYPE_BF16, device, device_id);
    _weights.out_embed = Tensor::create({voc, hs}, LLAISYS_DTYPE_BF16, device, device_id);
    _weights.out_norm_w = Tensor::create({hs}, LLAISYS_DTYPE_BF16, device, device_id);

    // Allocate per-layer weights
    _weights.attn_norm_w.resize(nlayer);
    _weights.attn_q_w.resize(nlayer);
    _weights.attn_q_b.resize(nlayer);
    _weights.attn_k_w.resize(nlayer);
    _weights.attn_k_b.resize(nlayer);
    _weights.attn_v_w.resize(nlayer);
    _weights.attn_v_b.resize(nlayer);
    _weights.attn_o_w.resize(nlayer);
    _weights.mlp_norm_w.resize(nlayer);
    _weights.mlp_gate_w.resize(nlayer);
    _weights.mlp_up_w.resize(nlayer);
    _weights.mlp_down_w.resize(nlayer);

    for (size_t i = 0; i < nlayer; i++) {
        _weights.attn_norm_w[i] = Tensor::create({hs}, LLAISYS_DTYPE_BF16, device, device_id);
        _weights.attn_q_w[i] = Tensor::create({nh * dh, hs}, LLAISYS_DTYPE_BF16, device, device_id);
        _weights.attn_q_b[i] = Tensor::create({nh * dh}, LLAISYS_DTYPE_BF16, device, device_id);
        _weights.attn_k_w[i] = Tensor::create({nkvh * dh, hs}, LLAISYS_DTYPE_BF16, device, device_id);
        _weights.attn_k_b[i] = Tensor::create({nkvh * dh}, LLAISYS_DTYPE_BF16, device, device_id);
        _weights.attn_v_w[i] = Tensor::create({nkvh * dh, hs}, LLAISYS_DTYPE_BF16, device, device_id);
        _weights.attn_v_b[i] = Tensor::create({nkvh * dh}, LLAISYS_DTYPE_BF16, device, device_id);
        _weights.attn_o_w[i] = Tensor::create({hs, nh * dh}, LLAISYS_DTYPE_BF16, device, device_id);
        _weights.mlp_norm_w[i] = Tensor::create({hs}, LLAISYS_DTYPE_BF16, device, device_id);
        _weights.mlp_gate_w[i] = Tensor::create({di, hs}, LLAISYS_DTYPE_BF16, device, device_id);
        _weights.mlp_up_w[i] = Tensor::create({di, hs}, LLAISYS_DTYPE_BF16, device, device_id);
        _weights.mlp_down_w[i] = Tensor::create({hs, di}, LLAISYS_DTYPE_BF16, device, device_id);
    }

    // Allocate KV cache
    _k_cache.resize(nlayer);
    _v_cache.resize(nlayer);
    for (size_t i = 0; i < nlayer; i++) {
        _k_cache[i] = Tensor::create({maxseq, nkvh, dh}, LLAISYS_DTYPE_BF16, device, device_id);
        _v_cache[i] = Tensor::create({maxseq, nkvh, dh}, LLAISYS_DTYPE_BF16, device, device_id);
    }

    // Allocate argmax output tensors
    _max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, device, device_id);
    _max_val = Tensor::create({1}, LLAISYS_DTYPE_BF16, device, device_id);
}

void Qwen2Model::allocateBuffers(size_t batch_size) {
    if (batch_size <= _max_batch) {
        return;
    }

    size_t hs = _meta.hs;
    size_t nh = _meta.nh;
    size_t nkvh = _meta.nkvh;
    size_t dh = _meta.dh;
    size_t di = _meta.di;
    size_t voc = _meta.voc;

    _hidden = Tensor::create({batch_size, hs}, LLAISYS_DTYPE_BF16, _device_type, _device_id);
    _residual = Tensor::create({batch_size, hs}, LLAISYS_DTYPE_BF16, _device_type, _device_id);
    _norm_out = Tensor::create({batch_size, hs}, LLAISYS_DTYPE_BF16, _device_type, _device_id);
    _q = Tensor::create({batch_size, nh * dh}, LLAISYS_DTYPE_BF16, _device_type, _device_id);
    _k = Tensor::create({batch_size, nkvh * dh}, LLAISYS_DTYPE_BF16, _device_type, _device_id);
    _v = Tensor::create({batch_size, nkvh * dh}, LLAISYS_DTYPE_BF16, _device_type, _device_id);
    _attn_out = Tensor::create({batch_size, nh, dh}, LLAISYS_DTYPE_BF16, _device_type, _device_id);
    _gate = Tensor::create({batch_size, di}, LLAISYS_DTYPE_BF16, _device_type, _device_id);
    _up = Tensor::create({batch_size, di}, LLAISYS_DTYPE_BF16, _device_type, _device_id);
    _mlp_out = Tensor::create({batch_size, hs}, LLAISYS_DTYPE_BF16, _device_type, _device_id);
    _logits = Tensor::create({1, voc}, LLAISYS_DTYPE_BF16, _device_type, _device_id);
    _pos_ids = Tensor::create({batch_size}, LLAISYS_DTYPE_I64, _device_type, _device_id);

    _max_batch = batch_size;
}

Qwen2Weights* Qwen2Model::weights() {
    return &_weights;
}

const Qwen2Meta* Qwen2Model::meta() const {
    return &_meta;
}

void Qwen2Model::resetCache() {
    _cache_len = 0;
}

int64_t Qwen2Model::infer(int64_t* token_ids, size_t ntoken) {
    // Set context to the correct device
    core::context().setDevice(_device_type, _device_id);

    allocateBuffers(ntoken);

    size_t nlayer = _meta.nlayer;
    size_t hs = _meta.hs;
    size_t nh = _meta.nh;
    size_t nkvh = _meta.nkvh;
    size_t dh = _meta.dh;
    float eps = _meta.epsilon;
    float theta = _meta.theta;
    float scale = 1.0f / std::sqrt(static_cast<float>(dh));

    // Create input tensor for token_ids
    auto input_ids = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, _device_type, _device_id);
    input_ids->load(token_ids);

    // Create position_ids: [cache_len, cache_len+1, ..., cache_len+ntoken-1]
    std::vector<int64_t> pos_data(ntoken);
    for (size_t i = 0; i < ntoken; i++) {
        pos_data[i] = static_cast<int64_t>(_cache_len + i);
    }
    auto pos_ids = _pos_ids->slice(0, 0, ntoken);
    pos_ids->load(pos_data.data());

    // Get views for current batch size
    auto hidden = _hidden->slice(0, 0, ntoken);
    auto residual = _residual->slice(0, 0, ntoken);
    auto norm_out = _norm_out->slice(0, 0, ntoken);
    auto q = _q->slice(0, 0, ntoken);
    auto k = _k->slice(0, 0, ntoken);
    auto v = _v->slice(0, 0, ntoken);
    auto attn_out = _attn_out->slice(0, 0, ntoken);
    auto gate = _gate->slice(0, 0, ntoken);
    auto up = _up->slice(0, 0, ntoken);

    // 1. Embedding lookup
    ops::embedding(hidden, input_ids, _weights.in_embed);

    // 2. Process each layer
    for (size_t layer = 0; layer < nlayer; layer++) {
        // Save residual
        ops::rearrange(residual, hidden);

        // Attention block
        // RMSNorm
        ops::rms_norm(norm_out, hidden, _weights.attn_norm_w[layer], eps);

        // Q/K/V projections
        ops::linear(q, norm_out, _weights.attn_q_w[layer], _weights.attn_q_b[layer]);
        ops::linear(k, norm_out, _weights.attn_k_w[layer], _weights.attn_k_b[layer]);
        ops::linear(v, norm_out, _weights.attn_v_w[layer], _weights.attn_v_b[layer]);

        // Reshape Q/K/V for attention: [ntoken, nh*dh] -> [ntoken, nh, dh]
        auto q_view = q->view({ntoken, nh, dh});
        auto k_view = k->view({ntoken, nkvh, dh});
        auto v_view = v->view({ntoken, nkvh, dh});

        // Apply RoPE
        ops::rope(q_view, q_view, pos_ids, theta);
        ops::rope(k_view, k_view, pos_ids, theta);

        // Update KV cache
        auto k_cache_slice = _k_cache[layer]->slice(0, _cache_len, _cache_len + ntoken);
        auto v_cache_slice = _v_cache[layer]->slice(0, _cache_len, _cache_len + ntoken);
        ops::rearrange(k_cache_slice, k_view);
        ops::rearrange(v_cache_slice, v_view);

        // Get full KV cache for attention
        auto k_full = _k_cache[layer]->slice(0, 0, _cache_len + ntoken);
        auto v_full = _v_cache[layer]->slice(0, 0, _cache_len + ntoken);

        // Self attention
        ops::self_attention(attn_out, q_view, k_full, v_full, scale);

        // Reshape attention output: [ntoken, nh, dh] -> [ntoken, hs]
        auto attn_out_flat = attn_out->view({ntoken, hs});

        // Output projection
        ops::linear(hidden, attn_out_flat, _weights.attn_o_w[layer], nullptr);

        // Add residual
        ops::add(hidden, hidden, residual);

        // MLP block
        // Save residual
        ops::rearrange(residual, hidden);

        // RMSNorm
        ops::rms_norm(norm_out, hidden, _weights.mlp_norm_w[layer], eps);

        // Gate and Up projections
        ops::linear(gate, norm_out, _weights.mlp_gate_w[layer], nullptr);
        ops::linear(up, norm_out, _weights.mlp_up_w[layer], nullptr);

        // SwiGLU
        ops::swiglu(gate, gate, up);

        // Down projection
        ops::linear(hidden, gate, _weights.mlp_down_w[layer], nullptr);

        // Add residual
        ops::add(hidden, hidden, residual);
    }

    // 3. Final RMSNorm
    ops::rms_norm(hidden, hidden, _weights.out_norm_w, eps);

    // 4. Get last token's hidden state and compute logits
    auto last_hidden = hidden->slice(0, ntoken - 1, ntoken);  // [1, hs]
    ops::linear(_logits, last_hidden, _weights.out_embed, nullptr);

    // 5. Argmax
    ops::argmax(_max_idx, _max_val, _logits);

    // 6. Update cache length
    _cache_len += ntoken;

    // 7. Read and return the result
    int64_t next_token;
    // Copy from device to host
    auto& runtime = core::context().runtime();
    runtime.api()->memcpy_sync(&next_token, _max_idx->data(), sizeof(int64_t), LLAISYS_MEMCPY_D2H);

    return next_token;
}

} // namespace llaisys::models
