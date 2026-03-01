#pragma once

#include "../../utils.hpp"
#include "../../tensor/tensor.hpp"
#include "qwen2_kvcache.hpp"

#include "llaisys/models/qwen2.h"

#include <vector>

namespace llaisys::models::qwen2 {

struct Qwen2Weights {
    tensor_t in_embed;
    tensor_t out_embed;
    tensor_t out_norm_w;
    std::vector<tensor_t> attn_norm_w;
    std::vector<tensor_t> attn_q_w;
    std::vector<tensor_t> attn_q_b;
    std::vector<tensor_t> attn_k_w;
    std::vector<tensor_t> attn_k_b;
    std::vector<tensor_t> attn_v_w;
    std::vector<tensor_t> attn_v_b;
    std::vector<tensor_t> attn_o_w;
    std::vector<tensor_t> mlp_norm_w;
    std::vector<tensor_t> mlp_gate_w;
    std::vector<tensor_t> mlp_up_w;
    std::vector<tensor_t> mlp_down_w;
};

class Qwen2Model {
public:
    Qwen2Model(const LlaisysQwen2Meta &meta, llaisysDeviceType_t device, int device_id);

    const LlaisysQwen2Meta &meta() const { return _meta; }
    llaisysDeviceType_t device() const { return _device; }
    int device_id() const { return _device_id; }

    void bind_weights(const LlaisysQwen2Weights &weights);
    int64_t infer(const int64_t *token_ids, size_t ntoken, int top_k, float top_p, float temperature, int64_t seed = -1);

private:
    void process_token(int64_t token_id);

    LlaisysQwen2Meta _meta;
    llaisysDeviceType_t _device;
    int _device_id;
    Qwen2Weights _weights;
    Qwen2KVCache _kv_cache;
    bool _weights_bound = false;

    tensor_t _token_ids;
    tensor_t _pos_ids;
    tensor_t _hidden;
    tensor_t _attn_norm;
    tensor_t _q_proj;
    tensor_t _k_proj;
    tensor_t _v_proj;
    tensor_t _q_view;
    tensor_t _k_view;
    tensor_t _v_view;
    tensor_t _q_rope;
    tensor_t _k_rope;
    tensor_t _attn_out;
    tensor_t _attn_out_flat;
    tensor_t _attn_proj;
    tensor_t _mlp_norm;
    tensor_t _mlp_gate;
    tensor_t _mlp_up;
    tensor_t _mlp_act;
    tensor_t _mlp_down;
    tensor_t _final_norm;
    tensor_t _logits;
    tensor_t _logits_flat;
    tensor_t _max_idx;
    tensor_t _max_val;

    float _attn_scale = 1.0f;
};

} // namespace llaisys::models::qwen2
