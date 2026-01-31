#pragma once

#include "llaisys/models/qwen2.h"
#include "../../tensor/tensor.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/add/op.hpp"

#include <vector>
#include <memory>

namespace llaisys {
namespace models {

struct Qwen2Model {
    LlaisysQwen2Meta meta;
    llaisysDeviceType_t device_type;
    int device_id;

    // Weights
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

    // KV Cache: [nlayer][max_seq, nkvh, dh]
    std::vector<tensor_t> k_cache;
    std::vector<tensor_t> v_cache;
    size_t cache_len;

    // Intermediate buffers
    tensor_t hidden;
    tensor_t hidden_norm;
    tensor_t q;
    tensor_t k;
    tensor_t v;
    tensor_t q_rope;
    tensor_t k_rope;
    tensor_t attn_out;
    tensor_t attn_proj;
    tensor_t gate;
    tensor_t up;
    tensor_t mlp_out;
    tensor_t logits;
    tensor_t max_idx;
    tensor_t max_val;
    tensor_t pos_ids;

    Qwen2Model(const LlaisysQwen2Meta* meta, llaisysDeviceType_t device, int device_id);
    ~Qwen2Model() = default;

    void allocateWeights();
    void allocateCache();
    void allocateBuffers(size_t max_seqlen);

    int64_t infer(int64_t* token_ids, size_t ntoken);
    void forward(size_t seqlen, size_t start_pos);
    void forwardLayer(size_t layer, size_t seqlen, size_t start_pos);
};

}  // namespace models
}  // namespace llaisys
