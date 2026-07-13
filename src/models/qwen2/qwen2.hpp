#pragma once

#include "../../tensor/tensor.hpp"
#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rearrange/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"

#include <vector>
#include <cmath>

namespace llaisys::models {

struct Qwen2Config {
    llaisysDataType_t dtype;
    size_t num_layers;      // nlayer: 28
    size_t hidden_size;     // hs: 1536
    size_t num_heads;       // nh: 12
    size_t num_kv_heads;    // nkvh: 2
    size_t head_dim;        // dh: 128 (hidden_size / num_heads)
    size_t intermediate_size; // di: 8960
    size_t max_seq_len;     // maxseq: 131072
    size_t vocab_size;      // voc: 151936
    float rms_norm_eps;     // epsilon: 1e-6
    float rope_theta;       // theta: 10000
    int64_t eos_token_id;   // end_token: 151643
};

class Qwen2Model {
private:
    Qwen2Config config_;
    llaisysDeviceType_t device_type_;
    int device_id_;
    size_t tp_rank_;
    size_t tp_world_size_;
    size_t hidden_shard_size_;
    size_t kv_hidden_size_;
    size_t kv_shard_size_;
    size_t intermediate_shard_size_;
    size_t vocab_shard_size_;
    size_t vocab_start_;
    size_t vocab_end_;
    
    // Weights
    tensor_t embed_tokens_;     // [vocab_shard_size, hidden_size]
    tensor_t lm_head_;          // [vocab_shard_size, hidden_size]
    tensor_t norm_weight_;      // [hidden_size]
    
    // Per-layer weights
    std::vector<tensor_t> input_layernorm_weight_;   // [hidden_size]
    std::vector<tensor_t> q_proj_weight_;            // [hidden_shard_size, hidden_size]
    std::vector<tensor_t> q_proj_bias_;              // [hidden_shard_size]
    std::vector<tensor_t> k_proj_weight_;            // [kv_shard_size, hidden_size]
    std::vector<tensor_t> k_proj_bias_;              // [kv_shard_size]
    std::vector<tensor_t> v_proj_weight_;            // [kv_shard_size, hidden_size]
    std::vector<tensor_t> v_proj_bias_;              // [kv_shard_size]
    std::vector<tensor_t> o_proj_weight_;            // [hidden_size, hidden_shard_size]
    std::vector<tensor_t> post_attn_layernorm_weight_; // [hidden_size]
    std::vector<tensor_t> gate_proj_weight_;         // [intermediate_shard_size, hidden_size]
    std::vector<tensor_t> up_proj_weight_;           // [intermediate_shard_size, hidden_size]
    std::vector<tensor_t> down_proj_weight_;         // [hidden_size, intermediate_shard_size]
    
    // KV Cache: local shard per rank [num_layers][max_seq_len, kv_shard_size]
    std::vector<tensor_t> k_cache_;
    std::vector<tensor_t> v_cache_;
    size_t cache_len_;  // Current cached sequence length
    
    // Intermediate buffers
    tensor_t hidden_states_;    // [seq_len, hidden_size]
    tensor_t normed_;           // [seq_len, hidden_size]
    tensor_t q_local_;          // [seq_len, hidden_shard_size]
    tensor_t k_local_;          // [seq_len, kv_shard_size]
    tensor_t v_local_;          // [seq_len, kv_shard_size]
    tensor_t attn_proj_;        // [seq_len, hidden_size]
    tensor_t gate_;             // [seq_len, intermediate_shard_size]
    tensor_t up_;               // [seq_len, intermediate_shard_size]
    tensor_t mlp_out_;          // [seq_len, intermediate_shard_size]
    tensor_t down_;             // [seq_len, hidden_size]
    tensor_t logits_local_;     // [1, vocab_shard_size]
    tensor_t pos_ids_;          // [seq_len]
    tensor_t input_ids_;        // [seq_len]
    tensor_t max_idx_;          // [1]
    tensor_t max_val_;          // [1]
    size_t buffer_seq_len_;
    
    void allocateBuffers(size_t seq_len);
    void allocateKVCache();
    
public:
    Qwen2Model(const Qwen2Config& config, llaisysDeviceType_t device_type, int device_id);
    ~Qwen2Model() = default;
    
    // Weight accessors for loading
    tensor_t& embedTokens() { return embed_tokens_; }
    tensor_t& lmHead() { return lm_head_; }
    tensor_t& normWeight() { return norm_weight_; }
    tensor_t& inputLayernormWeight(size_t layer) { return input_layernorm_weight_[layer]; }
    tensor_t& qProjWeight(size_t layer) { return q_proj_weight_[layer]; }
    tensor_t& qProjBias(size_t layer) { return q_proj_bias_[layer]; }
    tensor_t& kProjWeight(size_t layer) { return k_proj_weight_[layer]; }
    tensor_t& kProjBias(size_t layer) { return k_proj_bias_[layer]; }
    tensor_t& vProjWeight(size_t layer) { return v_proj_weight_[layer]; }
    tensor_t& vProjBias(size_t layer) { return v_proj_bias_[layer]; }
    tensor_t& oProjWeight(size_t layer) { return o_proj_weight_[layer]; }
    tensor_t& postAttnLayernormWeight(size_t layer) { return post_attn_layernorm_weight_[layer]; }
    tensor_t& gateProjWeight(size_t layer) { return gate_proj_weight_[layer]; }
    tensor_t& upProjWeight(size_t layer) { return up_proj_weight_[layer]; }
    tensor_t& downProjWeight(size_t layer) { return down_proj_weight_[layer]; }
    
    // Inference
    int64_t infer(const int64_t* token_ids, size_t num_tokens);
    
    // Reset KV cache
    void resetCache() { cache_len_ = 0; }
    
    const Qwen2Config& config() const { return config_; }
};

} // namespace llaisys::models
