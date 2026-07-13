#pragma once

#include "llaisys/models/qwen2.h"
#include "../../tensor/tensor.hpp"
#include <vector>

namespace llaisys::models::qwen2 {

class Qwen2Model {
public:
    Qwen2Model(const LlaisysQwen2Meta &meta, llaisysDeviceType_t device_type, int device_id);
    ~Qwen2Model();

    LlaisysQwen2Weights *weights();
    int64_t infer(int64_t *token_ids, size_t ntoken);

private:
    LlaisysQwen2Meta _meta;
    llaisysDeviceType_t _device_type;
    int _device_id;
    
    // 导出给 Python 用于加载数据的结构体
    LlaisysQwen2Weights _weights_export;
    
    // 权重张量存储 (保持 shared_ptr 引用)
    tensor_t _in_embed;
    tensor_t _out_embed;
    tensor_t _out_norm_w;
    
    std::vector<tensor_t> _layers_input_norm;
    std::vector<tensor_t> _layers_q_w;
    std::vector<tensor_t> _layers_q_b;
    std::vector<tensor_t> _layers_k_w;
    std::vector<tensor_t> _layers_k_b;
    std::vector<tensor_t> _layers_v_w;
    std::vector<tensor_t> _layers_v_b;
    std::vector<tensor_t> _layers_o_w;
    std::vector<tensor_t> _layers_post_norm;
    std::vector<tensor_t> _layers_gate_w;
    std::vector<tensor_t> _layers_up_w;
    std::vector<tensor_t> _layers_down_w;

    // KV Cache [layer][k/v] -> [max_seq, n_kv_head, head_dim]
    std::vector<tensor_t> _k_cache;
    std::vector<tensor_t> _v_cache;

    int64_t _current_pos;

    // 辅助函数：创建并初始化权重张量
    tensor_t create_weight(const std::vector<size_t>& shape);
};

} // namespace llaisys::models::qwen2