#pragma once

#include "../../tensor/tensor.hpp"
#include "llaisys.h"

#include <vector>

namespace llaisys::models {

struct Qwen2Meta {
    size_t nlayer;           // num_hidden_layers
    size_t hs;               // hidden_size
    size_t nh;               // num_attention_heads
    size_t nkvh;             // num_key_value_heads
    size_t dh;               // head_dim = hs / nh
    size_t di;               // intermediate_size
    size_t maxseq;           // max_position_embeddings
    size_t voc;              // vocab_size
    float epsilon;           // rms_norm_eps
    float theta;             // rope_theta
    int64_t end_token;       // eos_token_id
};

struct Qwen2Weights {
    tensor_t in_embed;       // [voc, hs]
    tensor_t out_embed;      // [voc, hs]
    tensor_t out_norm_w;     // [hs]

    // Per-layer weights
    std::vector<tensor_t> attn_norm_w;   // [nlayer][hs]
    std::vector<tensor_t> attn_q_w;      // [nlayer][nh*dh, hs]
    std::vector<tensor_t> attn_q_b;      // [nlayer][nh*dh]
    std::vector<tensor_t> attn_k_w;      // [nlayer][nkvh*dh, hs]
    std::vector<tensor_t> attn_k_b;      // [nlayer][nkvh*dh]
    std::vector<tensor_t> attn_v_w;      // [nlayer][nkvh*dh, hs]
    std::vector<tensor_t> attn_v_b;      // [nlayer][nkvh*dh]
    std::vector<tensor_t> attn_o_w;      // [nlayer][hs, nh*dh]

    std::vector<tensor_t> mlp_norm_w;    // [nlayer][hs]
    std::vector<tensor_t> mlp_gate_w;    // [nlayer][di, hs]
    std::vector<tensor_t> mlp_up_w;      // [nlayer][di, hs]
    std::vector<tensor_t> mlp_down_w;    // [nlayer][hs, di]
};

class Qwen2Model {
private:
    Qwen2Meta _meta;
    llaisysDeviceType_t _device_type;
    int _device_id;

    Qwen2Weights _weights;

    // KV Cache: [nlayer]
    std::vector<tensor_t> _k_cache;  // eaaxseq, nkvh, dh]
    std::vector<tensor_t> _v_cache;  // each: [maxseq, nkvh, dh]
    size_t _cache_len;

    // Intermediate buffers
    tensor_t _hidden;      // [seq, hs]
    tensor_t _residual;    // [seq, hs]
    tensor_t _norm_out;    // [seq, hs]
    tensor_t _q;           // [seq, nh*dh]
    tensor_t _k;           // [seq, nkvh*dh]
    tensor_t _v;           // [seq, nkvh*dh]
    tensor_t _attn_out;    // [seq, nh, dh]
    tensor_t _gate;        // [seq, di]
    tensor_t _up;          // [seq, di]
    tensor_t _mlp_out;     // [seq, hs]
    tensor_t _logits;      // [1, voc]
    tensor_t _pos_ids;     // [seq]
    tensor_t _max_idx;     // [1]
    tensor_t _max_val;     // [1]

    size_t _max_batch;     // Maximum batch size for intermediate buffers

    void allocateBuffers(size_t batch_size);

public:
    Qwen2Model(const Qwen2Meta* meta, llaisysDeviceType_t device, int device_id);
    ~Qwen2Model() = default;

    Qwen2Weights* weights();
    const Qwen2Meta* meta() const;

    int64_t infer(int64_t* token_ids, size_t ntoken);
    void resetCache();
};

} // namespace llaisys::models
