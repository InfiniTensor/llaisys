#pragma once

#include "../../core/llaisys_core.hpp"
#include "../../tensor/tensor.hpp"
#include "../../ops/ops.hpp"

#include <vector>
#include <memory>

namespace llaisys::models {

struct Qwen2Config {
    llaisysDataType_t dtype;
    size_t n_layers;      // nlayer
    size_t hidden_size;   // hs
    size_t n_heads;       // nh
    size_t n_kv_heads;    // nkvh
    size_t head_dim;      // dh
    size_t intermediate_size; // di
    size_t max_seq_len;   // maxseq
    size_t vocab_size;    // voc
    float rms_norm_eps;   // epsilon
    float rope_theta;     // theta
    int64_t eos_token_id; // end_token
};

struct Qwen2Weights {
    // Embedding
    tensor_t embed_tokens;

    // Output
    tensor_t lm_head;
    tensor_t norm_weight;

    // Per-layer weights
    std::vector<tensor_t> attn_norm_weight;
    std::vector<tensor_t> attn_q_weight;
    std::vector<tensor_t> attn_q_bias;
    std::vector<tensor_t> attn_k_weight;
    std::vector<tensor_t> attn_k_bias;
    std::vector<tensor_t> attn_v_weight;
    std::vector<tensor_t> attn_v_bias;
    std::vector<tensor_t> attn_o_weight;

    std::vector<tensor_t> mlp_norm_weight;
    std::vector<tensor_t> mlp_gate_weight;
    std::vector<tensor_t> mlp_up_weight;
    std::vector<tensor_t> mlp_down_weight;
};

struct KVCache {
    std::vector<tensor_t> k_cache;  // [n_layers]
    std::vector<tensor_t> v_cache;  // [n_layers]
    size_t current_seq_len;

    KVCache(size_t n_layers) : current_seq_len(0) {
        k_cache.resize(n_layers);
        v_cache.resize(n_layers);
    }
};

class Qwen2Model {
private:
    Qwen2Config config_;
    Qwen2Weights weights_;
    KVCache kv_cache_;
    llaisysDeviceType_t device_type_;
    int device_id_;

    // Forward pass for one layer
    tensor_t forward_layer(int layer_idx, tensor_t hidden_states, tensor_t position_ids);

    // Attention
    tensor_t forward_attention(int layer_idx, tensor_t hidden_states, tensor_t position_ids);

    // MLP
    tensor_t forward_mlp(int layer_idx, tensor_t hidden_states);

public:
    Qwen2Model(const Qwen2Config &config, llaisysDeviceType_t device_type, int device_id);

    ~Qwen2Model() = default;

    Qwen2Weights& weights() { return weights_; }

    const Qwen2Config& config() const { return config_; }

    // Forward pass: input_ids -> logits
    tensor_t forward(const std::vector<int64_t> &input_ids);

    // Generate next token (argmax)
    int64_t generate_next_token(const std::vector<int64_t> &input_ids);

    // Reset KV cache
    void reset_cache();
};

} // namespace llaisys::models
