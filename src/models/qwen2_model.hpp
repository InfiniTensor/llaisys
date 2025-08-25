#pragma once

#include "../tensor/tensor.hpp"
#include "../core/llaisys_core.hpp"
#include "llaisys/models/qwen2.h"
#include "../llaisys/llaisys_tensor.hpp"

#include <vector>
#include <memory>

namespace llaisys::models {

class Qwen2Model {
private:
    LlaisysQwen2Meta _meta;
    llaisysDeviceType_t _device_type;
    int _device_id;
    
    // 模型权重
    std::unique_ptr<LlaisysQwen2Weights> _weights;
    
    // KV缓存相关
    std::vector<tensor_t> _k_cache;  // [nlayer]
    std::vector<tensor_t> _v_cache;  // [nlayer] 
    size_t _cache_len;  // 当前缓存长度
    
    // 工作张量（复用以减少内存分配）
    tensor_t _input_ids;
    tensor_t _positions; 
    tensor_t _hidden_states;
    tensor_t _residual;
    tensor_t _norm_output;
    tensor_t _attn_output;
    tensor_t _mlp_output;
    tensor_t _q, _k, _v;
    tensor_t _gate, _up;
    tensor_t _logits;
    
public:
    Qwen2Model(const LlaisysQwen2Meta* meta, llaisysDeviceType_t device_type, int device_id);
    ~Qwen2Model();
    
    // 获取权重结构
    LlaisysQwen2Weights* weights() { return _weights.get(); }
    
    // 推理一个token
    int64_t infer(int64_t* token_ids, size_t ntoken);
    
private:
    void _allocate_tensors();
    void _allocate_weights();
    tensor_t _forward_layer(tensor_t hidden_states, size_t layer_idx, size_t seq_len, size_t start_pos);
    void _apply_rope(tensor_t q, tensor_t k, size_t start_pos, size_t seq_len);
    tensor_t _apply_attention(tensor_t q, tensor_t k, tensor_t v, size_t layer_idx, size_t seq_len, size_t start_pos);
    int64_t _sample_token(tensor_t logits);
    
    // 辅助函数：将C类型llaisysTensor_t转换为C++类型tensor_t
    static tensor_t _to_tensor(llaisysTensor_t t) {
        return t ? reinterpret_cast<LlaisysTensor*>(t)->tensor : nullptr;
    }
};

} // namespace llaisys::models
