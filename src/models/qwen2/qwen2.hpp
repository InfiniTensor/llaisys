#pragma once
#include "llaisys/models/qwen2.h"
#include "../../tensor/tensor.hpp"
#include "llaisys/ops.h"
#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../ops/rearrange/op.hpp" 
#include <vector>

namespace llaisys::models::qwen2 {

class Qwen2Model {
public:
    LlaisysQwen2Meta _meta;
    LlaisysQwen2Weights _weights_ptr; // 用于传递给 Python 的指针结构
    
    // 实际存储权重的容器
    tensor_t in_embed;
    tensor_t out_embed;
    tensor_t out_norm_w;
    
    // Layers weights (Vector of tensors)
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

    // KV Cache
    std::vector<tensor_t> k_cache;
    std::vector<tensor_t> v_cache;
    size_t _current_pos = 0;

    Qwen2Model(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int device_id);
    ~Qwen2Model() = default;

    int64_t forward(const int64_t *token_ids, size_t ntoken);
    
private:
    void init_weights(llaisysDeviceType_t device, int device_id);
    void init_kv_cache(llaisysDeviceType_t device, int device_id);
};

}