#pragma once
#include "../../tensor/tensor.hpp"
#include "llaisys/models/qwen2.h"
#include <vector>

namespace llaisys::models {

class Qwen2Model {
private:
    LlaisysQwen2Meta meta;
    llaisysDeviceType_t device_type;
    int device_id;
    
    tensor_t in_embed, out_embed, out_norm_w;
    std::vector<tensor_t> attn_norm_w, attn_q_w, attn_q_b, attn_k_w, attn_k_b, attn_v_w, attn_v_b, attn_o_w;
    std::vector<tensor_t> mlp_norm_w, mlp_gate_w, mlp_up_w, mlp_down_w;
    std::vector<tensor_t> k_cache, v_cache;
    size_t cur_seq_len;
    
public:
    Qwen2Model(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int device_id);
    LlaisysQwen2Weights getWeights();
    int64_t infer(int64_t *token_ids, size_t ntoken);
};

}
