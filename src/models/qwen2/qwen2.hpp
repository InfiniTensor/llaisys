#pragma once
#include "llaisys/models/qwen2.h"
#include "../../tensor/tensor.hpp"
#include <vector>
#include <memory> 

namespace llaisys {

class Qwen2Model {
public:
    Qwen2Model(const LlaisysQwen2Meta& meta, llaisysDeviceType_t device, int device_id);
    ~Qwen2Model();

    LlaisysQwen2Weights* getWeights() { return &_weights; }
    
    int64_t infer(const int64_t* token_ids, size_t ntoken);

private:
    LlaisysQwen2Meta _meta;
    LlaisysQwen2Weights _weights;
    
    llaisysDeviceType_t _device_type;
    int _device_id;

    struct KVCache {
        tensor_t k;
        tensor_t v;
    };
    std::vector<KVCache> _kv_caches;

    size_t _cur_pos = 0;

    tensor_t _hidden_states;
    tensor_t _residual;
    tensor_t _ln_out;
    tensor_t _attn_out;      
    tensor_t _mlp_out;
    tensor_t _logits;

    tensor_t _tokens_tensor;
    tensor_t _pos_ids;

    void init_buffers();
    
    std::vector<tensor_t> _layers_attn_norm_w;
    std::vector<tensor_t> _layers_attn_q_w;
    std::vector<tensor_t> _layers_attn_q_b;
    std::vector<tensor_t> _layers_attn_k_w;
    std::vector<tensor_t> _layers_attn_k_b;
    std::vector<tensor_t> _layers_attn_v_w;
    std::vector<tensor_t> _layers_attn_v_b;
    std::vector<tensor_t> _layers_attn_o_w;
    std::vector<tensor_t> _layers_mlp_norm_w;
    std::vector<tensor_t> _layers_mlp_gate_w;
    std::vector<tensor_t> _layers_mlp_up_w;
    std::vector<tensor_t> _layers_mlp_down_w;

    void allocate_layers_weights();
};

} // namespace llaisys
