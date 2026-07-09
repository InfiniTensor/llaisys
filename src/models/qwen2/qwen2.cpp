#include "qwen2.hpp"
#include <iostream>
#include <cstring>
#include <cmath>

namespace llaisys::models {

Qwen2Model::Qwen2Model(const Qwen2Config &config, llaisysDeviceType_t device_type, int device_id)
    : config_(config), kv_cache_(config.n_layers), device_type_(device_type), device_id_(device_id) {

    core::context().setDevice(device_type_, device_id_);

    // Initialize embedding weights
    weights_.embed_tokens = Tensor::create({config.vocab_size, config.hidden_size}, config.dtype, device_type, device_id);
    weights_.lm_head = Tensor::create({config.vocab_size, config.hidden_size}, config.dtype, device_type, device_id);
    weights_.norm_weight = Tensor::create({config.hidden_size}, config.dtype, device_type, device_id);

    // Initialize per-layer weights
    weights_.attn_norm_weight.resize(config.n_layers);
    weights_.attn_q_weight.resize(config.n_layers);
    weights_.attn_q_bias.resize(config.n_layers);
    weights_.attn_k_weight.resize(config.n_layers);
    weights_.attn_k_bias.resize(config.n_layers);
    weights_.attn_v_weight.resize(config.n_layers);
    weights_.attn_v_bias.resize(config.n_layers);
    weights_.attn_o_weight.resize(config.n_layers);

    weights_.mlp_norm_weight.resize(config.n_layers);
    weights_.mlp_gate_weight.resize(config.n_layers);
    weights_.mlp_up_weight.resize(config.n_layers);
    weights_.mlp_down_weight.resize(config.n_layers);

    for (size_t i = 0; i < config.n_layers; i++) {
        // Attention weights
        weights_.attn_norm_weight[i] = Tensor::create({config.hidden_size}, config.dtype, device_type, device_id);
        weights_.attn_q_weight[i] = Tensor::create({config.hidden_size, config.hidden_size}, config.dtype, device_type, device_id);
        weights_.attn_q_bias[i] = Tensor::create({config.hidden_size}, config.dtype, device_type, device_id);
        weights_.attn_k_weight[i] = Tensor::create({config.n_kv_heads * config.head_dim, config.hidden_size}, config.dtype, device_type, device_id);
        weights_.attn_k_bias[i] = Tensor::create({config.n_kv_heads * config.head_dim}, config.dtype, device_type, device_id);
        weights_.attn_v_weight[i] = Tensor::create({config.n_kv_heads * config.head_dim, config.hidden_size}, config.dtype, device_type, device_id);
        weights_.attn_v_bias[i] = Tensor::create({config.n_kv_heads * config.head_dim}, config.dtype, device_type, device_id);
        weights_.attn_o_weight[i] = Tensor::create({config.hidden_size, config.hidden_size}, config.dtype, device_type, device_id);

        // MLP weights
        weights_.mlp_norm_weight[i] = Tensor::create({config.hidden_size}, config.dtype, device_type, device_id);
        weights_.mlp_gate_weight[i] = Tensor::create({config.intermediate_size, config.hidden_size}, config.dtype, device_type, device_id);
        weights_.mlp_up_weight[i] = Tensor::create({config.intermediate_size, config.hidden_size}, config.dtype, device_type, device_id);
        weights_.mlp_down_weight[i] = Tensor::create({config.hidden_size, config.intermediate_size}, config.dtype, device_type, device_id);
    }
}

void Qwen2Model::reset_cache() {
    kv_cache_.current_seq_len = 0;
    for (auto &k : kv_cache_.k_cache) {
        k = nullptr;
    }
    for (auto &v : kv_cache_.v_cache) {
        v = nullptr;
    }
}

tensor_t Qwen2Model::forward_attention(int layer_idx, tensor_t hidden_states, tensor_t position_ids) {
    // hidden_states: [seq_len, hidden_size]
    size_t seq_len = hidden_states->shape()[0];

    // Input norm
    auto normed = Tensor::create({seq_len, config_.hidden_size}, config_.dtype, device_type_, device_id_);
    ops::rms_norm(normed, hidden_states, weights_.attn_norm_weight[layer_idx], config_.rms_norm_eps);

    // Q, K, V projections
    auto q = Tensor::create({seq_len, config_.hidden_size}, config_.dtype, device_type_, device_id_);
    auto k = Tensor::create({seq_len, config_.n_kv_heads * config_.head_dim}, config_.dtype, device_type_, device_id_);
    auto v = Tensor::create({seq_len, config_.n_kv_heads * config_.head_dim}, config_.dtype, device_type_, device_id_);

    ops::linear(q, normed, weights_.attn_q_weight[layer_idx], weights_.attn_q_bias[layer_idx]);
    ops::linear(k, normed, weights_.attn_k_weight[layer_idx], weights_.attn_k_bias[layer_idx]);
    ops::linear(v, normed, weights_.attn_v_weight[layer_idx], weights_.attn_v_bias[layer_idx]);

    // Reshape to [seq_len, n_heads, head_dim]
    auto q_reshaped = q->view({seq_len, config_.n_heads, config_.head_dim});
    auto k_reshaped = k->view({seq_len, config_.n_kv_heads, config_.head_dim});
    auto v_reshaped = v->view({seq_len, config_.n_kv_heads, config_.head_dim});

    // Apply RoPE
    auto q_rope = Tensor::create({seq_len, config_.n_heads, config_.head_dim}, config_.dtype, device_type_, device_id_);
    auto k_rope = Tensor::create({seq_len, config_.n_kv_heads, config_.head_dim}, config_.dtype, device_type_, device_id_);

    ops::rope(q_rope, q_reshaped, position_ids, config_.rope_theta);
    ops::rope(k_rope, k_reshaped, position_ids, config_.rope_theta);

    // Update KV cache
    tensor_t k_full, v_full;
    if (kv_cache_.k_cache[layer_idx] == nullptr) {
        // First iteration
        k_full = k_rope;
        v_full = v_reshaped;
        kv_cache_.k_cache[layer_idx] = k_rope;
        kv_cache_.v_cache[layer_idx] = v_reshaped;
    } else {
        // Concat with previous cache
        size_t prev_len = kv_cache_.k_cache[layer_idx]->shape()[0];
        size_t total_len = prev_len + seq_len;

        k_full = Tensor::create({total_len, config_.n_kv_heads, config_.head_dim}, config_.dtype, device_type_, device_id_);
        v_full = Tensor::create({total_len, config_.n_kv_heads, config_.head_dim}, config_.dtype, device_type_, device_id_);

        // Copy previous cache
        auto k_prev_slice = k_full->slice(0, 0, prev_len);
        auto k_new_slice = k_full->slice(0, prev_len, total_len);
        auto v_prev_slice = v_full->slice(0, 0, prev_len);
        auto v_new_slice = v_full->slice(0, prev_len, total_len);

        // Manual copy (since we don't have a copy operator)
        size_t elem_size = config_.n_kv_heads * config_.head_dim;
        memcpy(k_prev_slice->data(), kv_cache_.k_cache[layer_idx]->data(),
                    prev_len * elem_size * k_prev_slice->elementSize());
        memcpy(k_new_slice->data(), k_rope->data(),
                    seq_len * elem_size * k_new_slice->elementSize());
        memcpy(v_prev_slice->data(), kv_cache_.v_cache[layer_idx]->data(),
                    prev_len * elem_size * v_prev_slice->elementSize());
        memcpy(v_new_slice->data(), v_reshaped->data(),
                    seq_len * elem_size * v_new_slice->elementSize());

        kv_cache_.k_cache[layer_idx] = k_full;
        kv_cache_.v_cache[layer_idx] = v_full;
    }

    // Self-attention
    auto attn_output = Tensor::create({seq_len, config_.n_heads, config_.head_dim}, config_.dtype, device_type_, device_id_);
    float scale = 1.0f / std::sqrt(static_cast<float>(config_.head_dim));

    ops::self_attention(attn_output, q_rope, k_full, v_full, scale);

    // Reshape back: [seq_len, n_heads, head_dim] -> [seq_len, hidden_size]
    auto attn_flat = attn_output->view({seq_len, config_.hidden_size});

    // Output projection
    auto output = Tensor::create({seq_len, config_.hidden_size}, config_.dtype, device_type_, device_id_);
    ops::linear(output, attn_flat, weights_.attn_o_weight[layer_idx], nullptr);

    // Residual connection
    auto result = Tensor::create({seq_len, config_.hidden_size}, config_.dtype, device_type_, device_id_);
    ops::add(result, hidden_states, output);

    return result;
}

tensor_t Qwen2Model::forward_mlp(int layer_idx, tensor_t hidden_states) {
    // hidden_states: [seq_len, hidden_size]
    size_t seq_len = hidden_states->shape()[0];

    // Post-attention norm
    auto normed = Tensor::create({seq_len, config_.hidden_size}, config_.dtype, device_type_, device_id_);
    ops::rms_norm(normed, hidden_states, weights_.mlp_norm_weight[layer_idx], config_.rms_norm_eps);

    // Gate and Up projections
    auto gate = Tensor::create({seq_len, config_.intermediate_size}, config_.dtype, device_type_, device_id_);
    auto up = Tensor::create({seq_len, config_.intermediate_size}, config_.dtype, device_type_, device_id_);

    ops::linear(gate, normed, weights_.mlp_gate_weight[layer_idx], nullptr);
    ops::linear(up, normed, weights_.mlp_up_weight[layer_idx], nullptr);

    // SwiGLU activation
    auto activated = Tensor::create({seq_len, config_.intermediate_size}, config_.dtype, device_type_, device_id_);
    ops::swiglu(activated, gate, up);

    // Down projection
    auto mlp_output = Tensor::create({seq_len, config_.hidden_size}, config_.dtype, device_type_, device_id_);
    ops::linear(mlp_output, activated, weights_.mlp_down_weight[layer_idx], nullptr);

    // Residual connection
    auto result = Tensor::create({seq_len, config_.hidden_size}, config_.dtype, device_type_, device_id_);
    ops::add(result, hidden_states, mlp_output);

    return result;
}

tensor_t Qwen2Model::forward_layer(int layer_idx, tensor_t hidden_states, tensor_t position_ids) {
    // Attention block
    auto attn_output = forward_attention(layer_idx, hidden_states, position_ids);

    // MLP block
    auto mlp_output = forward_mlp(layer_idx, attn_output);

    return mlp_output;
}

tensor_t Qwen2Model::forward(const std::vector<int64_t> &input_ids) {
    size_t seq_len = input_ids.size();

    core::context().setDevice(device_type_, device_id_);

    // Create input tensor
    auto input_tensor = Tensor::create({seq_len}, LLAISYS_DTYPE_I64, device_type_, device_id_);
    input_tensor->load(input_ids.data());

    // Embedding
    auto hidden_states = Tensor::create({seq_len, config_.hidden_size}, config_.dtype, device_type_, device_id_);
    ops::embedding(hidden_states, input_tensor, weights_.embed_tokens);

    // Create position IDs
    std::vector<int64_t> pos_ids(seq_len);
    for (size_t i = 0; i < seq_len; i++) {
        pos_ids[i] = kv_cache_.current_seq_len + i;
    }
    auto position_ids = Tensor::create({seq_len}, LLAISYS_DTYPE_I64, device_type_, device_id_);
    position_ids->load(pos_ids.data());

    // Forward through all layers
    for (size_t layer = 0; layer < config_.n_layers; layer++) {
        hidden_states = forward_layer(layer, hidden_states, position_ids);
    }

    // Final norm
    auto normed = Tensor::create({seq_len, config_.hidden_size}, config_.dtype, device_type_, device_id_);
    ops::rms_norm(normed, hidden_states, weights_.norm_weight, config_.rms_norm_eps);

    // LM head (only need last token for generation)
    auto last_hidden = normed->slice(0, seq_len - 1, seq_len);
    auto last_hidden_2d = last_hidden->view({1, config_.hidden_size});

    auto logits = Tensor::create({1, config_.vocab_size}, config_.dtype, device_type_, device_id_);
    ops::linear(logits, last_hidden_2d, weights_.lm_head, nullptr);

    // Update cache length
    kv_cache_.current_seq_len += seq_len;

    return logits;
}

int64_t Qwen2Model::generate_next_token(const std::vector<int64_t> &input_ids) {
    auto logits = forward(input_ids);

    // Argmax to get next token
    auto logits_1d = logits->view({config_.vocab_size});
    auto max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, device_type_, device_id_);
    auto max_val = Tensor::create({1}, config_.dtype, device_type_, device_id_);

    ops::argmax(max_idx, max_val, logits_1d);

    // Read result
    int64_t next_token;
    if (device_type_ == LLAISYS_DEVICE_CPU) {
        next_token = *reinterpret_cast<int64_t*>(max_idx->data());
    } else {
        // Copy from device to host
        core::context().runtime().api()->memcpy_sync(
            &next_token, max_idx->data(), sizeof(int64_t), LLAISYS_MEMCPY_D2H);
    }

    return next_token;
}

} // namespace llaisys::models
