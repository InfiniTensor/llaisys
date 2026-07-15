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
    int64_t generateNext(const int64_t* token_ids, size_t ntoken, int top_k, float top_p, float temperature);
    void reset();
    void truncate(size_t position);

private:
    LlaisysQwen2Meta _meta;
    LlaisysQwen2Weights _weights;
    
    llaisysDeviceType_t _device_type;
    int _device_id;

    struct KVCache {
        tensor_t k;
        tensor_t v;
    };
    struct DecodeLayerBuffers {
        tensor_t q_flat;
        tensor_t k_flat;
        tensor_t v_flat;
        tensor_t q_view;
        tensor_t k_view;
        tensor_t v_view;
        tensor_t attn_val_3d;
        tensor_t attn_val_2d;
        tensor_t gate;
        tensor_t up;
        tensor_t swiglu;
    };
    struct PrefillBuffers {
        size_t capacity = 0;
        tensor_t tokens;
        tensor_t hidden_states;
        tensor_t normed;
        tensor_t q_flat;
        tensor_t k_flat;
        tensor_t v_flat;
        tensor_t attn_val_3d;
        tensor_t attn_out;
        tensor_t gate;
        tensor_t up;
        tensor_t swiglu;
        tensor_t mlp_out;
    };
    std::vector<KVCache> _kv_caches;
    std::vector<DecodeLayerBuffers> _decode_layers;
    PrefillBuffers _prefill;

    size_t _cur_pos = 0;

    tensor_t _hidden_states;
    tensor_t _ln_out;
    tensor_t _attn_out;
    tensor_t _mlp_out;
    tensor_t _logits;
    tensor_t _tokens_tensor;
    tensor_t _single_pos_id;
    tensor_t _pos_ids;
    tensor_t _argmax_idx;
    tensor_t _argmax_val;

    void init_buffers();
    void ensure_prefill_buffers(size_t ntoken);
    tensor_t forward(const int64_t* token_ids, size_t ntoken);
    tensor_t forward_single_token(const int64_t* token_id);
};

} // namespace llaisys
