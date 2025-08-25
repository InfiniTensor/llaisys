#include "qwen2_model.hpp"
#include "../ops/add/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"
#include "../ops/argmax/op.hpp"

#include <cmath>
#include <random>

namespace llaisys::models {

Qwen2Model::Qwen2Model(const LlaisysQwen2Meta* meta, llaisysDeviceType_t device_type, int device_id)
    : _meta(*meta), _device_type(device_type), _device_id(device_id), _cache_len(0) {
    
    // 设置设备上下文
    core::context().setDevice(device_type, device_id);
    
    // 分配权重和张量
    _allocate_weights();
    _allocate_tensors();
}

Qwen2Model::~Qwen2Model() {
    // 智能指针和tensor_t会自动清理
}

void Qwen2Model::_allocate_weights() {
    _weights = std::make_unique<LlaisysQwen2Weights>();
    
    // 分配主要权重
    _weights->in_embed = nullptr;     // 输入embedding
    _weights->out_embed = nullptr;    // 输出embedding
    _weights->out_norm_w = nullptr;   // 最终层归一化
    
    // 分配每层权重数组
    _weights->attn_norm_w = new llaisysTensor_t[_meta.nlayer];
    _weights->attn_q_w = new llaisysTensor_t[_meta.nlayer];
    _weights->attn_q_b = new llaisysTensor_t[_meta.nlayer];
    _weights->attn_k_w = new llaisysTensor_t[_meta.nlayer];
    _weights->attn_k_b = new llaisysTensor_t[_meta.nlayer];
    _weights->attn_v_w = new llaisysTensor_t[_meta.nlayer];
    _weights->attn_v_b = new llaisysTensor_t[_meta.nlayer];
    _weights->attn_o_w = new llaisysTensor_t[_meta.nlayer];
    _weights->mlp_norm_w = new llaisysTensor_t[_meta.nlayer];
    _weights->mlp_gate_w = new llaisysTensor_t[_meta.nlayer];
    _weights->mlp_up_w = new llaisysTensor_t[_meta.nlayer];
    _weights->mlp_down_w = new llaisysTensor_t[_meta.nlayer];
    
    // 初始化为nullptr
    for (size_t i = 0; i < _meta.nlayer; i++) {
        _weights->attn_norm_w[i] = nullptr;
        _weights->attn_q_w[i] = nullptr;
        _weights->attn_q_b[i] = nullptr;
        _weights->attn_k_w[i] = nullptr;
        _weights->attn_k_b[i] = nullptr;
        _weights->attn_v_w[i] = nullptr;
        _weights->attn_v_b[i] = nullptr;
        _weights->attn_o_w[i] = nullptr;
        _weights->mlp_norm_w[i] = nullptr;
        _weights->mlp_gate_w[i] = nullptr;
        _weights->mlp_up_w[i] = nullptr;
        _weights->mlp_down_w[i] = nullptr;
    }
}

void Qwen2Model::_allocate_tensors() {
    using namespace llaisys;
    
    // 输入相关张量
    _input_ids = Tensor::create({_meta.maxseq}, LLAISYS_DTYPE_I64, _device_type, _device_id);
    _positions = Tensor::create({_meta.maxseq}, LLAISYS_DTYPE_I64, _device_type, _device_id);
    
    // 隐藏状态张量
    _hidden_states = Tensor::create({_meta.maxseq, _meta.hs}, _meta.dtype, _device_type, _device_id);
    _residual = Tensor::create({_meta.maxseq, _meta.hs}, _meta.dtype, _device_type, _device_id);
    _norm_output = Tensor::create({_meta.maxseq, _meta.hs}, _meta.dtype, _device_type, _device_id);
    _attn_output = Tensor::create({_meta.maxseq, _meta.hs}, _meta.dtype, _device_type, _device_id);
    _mlp_output = Tensor::create({_meta.maxseq, _meta.hs}, _meta.dtype, _device_type, _device_id);
    
    // Attention张量
    _q = Tensor::create({_meta.maxseq, _meta.nh, _meta.dh}, _meta.dtype, _device_type, _device_id);
    _k = Tensor::create({_meta.maxseq, _meta.nkvh, _meta.dh}, _meta.dtype, _device_type, _device_id);
    _v = Tensor::create({_meta.maxseq, _meta.nkvh, _meta.dh}, _meta.dtype, _device_type, _device_id);
    
    // MLP张量
    _gate = Tensor::create({_meta.maxseq, _meta.di}, _meta.dtype, _device_type, _device_id);
    _up = Tensor::create({_meta.maxseq, _meta.di}, _meta.dtype, _device_type, _device_id);
    
    // 输出张量
    _logits = Tensor::create({_meta.voc}, _meta.dtype, _device_type, _device_id);
    
    // KV缓存
    _k_cache.resize(_meta.nlayer);
    _v_cache.resize(_meta.nlayer);
    for (size_t i = 0; i < _meta.nlayer; i++) {
        _k_cache[i] = Tensor::create({_meta.maxseq, _meta.nkvh, _meta.dh}, _meta.dtype, _device_type, _device_id);
        _v_cache[i] = Tensor::create({_meta.maxseq, _meta.nkvh, _meta.dh}, _meta.dtype, _device_type, _device_id);
    }
}

int64_t Qwen2Model::infer(int64_t* token_ids, size_t ntoken) {
    using namespace llaisys;
    
    // 设置设备上下文
    core::context().setDevice(_device_type, _device_id);
    
    size_t start_pos = _cache_len;
    size_t seq_len = ntoken;
    
    // 加载输入tokens
    auto input_slice = _input_ids->slice(0, start_pos, start_pos + seq_len);
    input_slice->load(token_ids);
    
    // 生成位置编码
    std::vector<int64_t> positions(seq_len);
    for (size_t i = 0; i < seq_len; i++) {
        positions[i] = static_cast<int64_t>(start_pos + i);
    }
    auto pos_slice = _positions->slice(0, start_pos, start_pos + seq_len);
    pos_slice->load(positions.data());
    
    // 1. Input Embedding
    auto hidden_slice = _hidden_states->slice(0, start_pos, start_pos + seq_len);
    ops::embedding(hidden_slice, input_slice, _to_tensor(_weights->in_embed));
    
    // 2. Transformer Layers
    for (size_t layer = 0; layer < _meta.nlayer; layer++) {
        hidden_slice = _forward_layer(hidden_slice, layer, seq_len, start_pos);
    }
    
    // 3. Final Layer Norm
    ops::rms_norm(_norm_output->slice(0, start_pos, start_pos + seq_len), 
                  hidden_slice, _to_tensor(_weights->out_norm_w), _meta.epsilon);
    
    // 4. Output Projection (只需要最后一个token的logits)
    auto last_hidden = _norm_output->slice(0, start_pos + seq_len - 1, start_pos + seq_len);
    auto last_hidden_2d = last_hidden->view({1, _meta.hs});
    auto logits_2d = _logits->view({1, _meta.voc});
    
    ops::linear(logits_2d, last_hidden_2d, _to_tensor(_weights->out_embed), nullptr);
    
    // 5. Sample next token
    int64_t next_token = _sample_token(_logits);
    
    // 更新缓存长度
    _cache_len += seq_len;
    
    return next_token;
}

tensor_t Qwen2Model::_forward_layer(tensor_t hidden_states, size_t layer_idx, size_t seq_len, size_t start_pos) {
    using namespace llaisys;
    
    // 1. Attention Layer Norm
    ops::rms_norm(_norm_output->slice(0, start_pos, start_pos + seq_len), 
                  hidden_states, _to_tensor(_weights->attn_norm_w[layer_idx]), _meta.epsilon);
    
    // 2. Q, K, V projections
    auto q_slice = _q->slice(0, 0, seq_len);
    auto k_slice = _k->slice(0, 0, seq_len);  
    auto v_slice = _v->slice(0, 0, seq_len);
    auto norm_slice = _norm_output->slice(0, start_pos, start_pos + seq_len);
    
    ops::linear(q_slice->view({seq_len, _meta.hs}), norm_slice->view({seq_len, _meta.hs}),
               _to_tensor(_weights->attn_q_w[layer_idx]), _to_tensor(_weights->attn_q_b[layer_idx]));
    
    ops::linear(k_slice->view({seq_len, _meta.hs}), norm_slice->view({seq_len, _meta.hs}), 
               _to_tensor(_weights->attn_k_w[layer_idx]), _to_tensor(_weights->attn_k_b[layer_idx]));
    
    ops::linear(v_slice->view({seq_len, _meta.hs}), norm_slice->view({seq_len, _meta.hs}),
               _to_tensor(_weights->attn_v_w[layer_idx]), _to_tensor(_weights->attn_v_b[layer_idx]));
    
    // 3. Apply RoPE
    _apply_rope(q_slice, k_slice, start_pos, seq_len);
    
    // 4. Self-Attention
    auto attn_out = _apply_attention(q_slice, k_slice, v_slice, layer_idx, seq_len, start_pos);
    
    // 5. Attention output projection
    ops::linear(_attn_output->slice(0, start_pos, start_pos + seq_len)->view({seq_len, _meta.hs}),
               attn_out->view({seq_len, _meta.hs}), _to_tensor(_weights->attn_o_w[layer_idx]), nullptr);
    
    // 6. Residual connection
    ops::add(_residual->slice(0, start_pos, start_pos + seq_len), 
             hidden_states, _attn_output->slice(0, start_pos, start_pos + seq_len));
    
    // 7. MLP Layer Norm
    ops::rms_norm(_norm_output->slice(0, start_pos, start_pos + seq_len),
                  _residual->slice(0, start_pos, start_pos + seq_len),
                  _to_tensor(_weights->mlp_norm_w[layer_idx]), _meta.epsilon);
    
    // 8. MLP
    auto gate_slice = _gate->slice(0, 0, seq_len);
    auto up_slice = _up->slice(0, 0, seq_len);
    
    ops::linear(gate_slice->view({seq_len, _meta.di}), norm_slice->view({seq_len, _meta.hs}),
               _to_tensor(_weights->mlp_gate_w[layer_idx]), nullptr);
    
    ops::linear(up_slice->view({seq_len, _meta.di}), norm_slice->view({seq_len, _meta.hs}),
               _to_tensor(_weights->mlp_up_w[layer_idx]), nullptr);
    
    ops::swiglu(_mlp_output->slice(0, 0, seq_len)->view({seq_len, _meta.di}), 
                gate_slice->view({seq_len, _meta.di}), up_slice->view({seq_len, _meta.di}));
    
    ops::linear(_norm_output->slice(0, start_pos, start_pos + seq_len)->view({seq_len, _meta.hs}),
               _mlp_output->slice(0, 0, seq_len)->view({seq_len, _meta.di}),
               _to_tensor(_weights->mlp_down_w[layer_idx]), nullptr);
    
    // 9. Final residual connection
    ops::add(hidden_states,
             _residual->slice(0, start_pos, start_pos + seq_len),
             _norm_output->slice(0, start_pos, start_pos + seq_len));
    
    return hidden_states;
}

void Qwen2Model::_apply_rope(tensor_t q, tensor_t k, size_t start_pos, size_t seq_len) {
    using namespace llaisys;
    
    // 生成位置序列
    std::vector<int64_t> positions(seq_len);
    for (size_t i = 0; i < seq_len; i++) {
        positions[i] = static_cast<int64_t>(start_pos + i);
    }
    
    auto pos_tensor = Tensor::create({seq_len}, LLAISYS_DTYPE_I64, _device_type, _device_id);
    pos_tensor->load(positions.data());
    
    ops::rope(q, q, pos_tensor, _meta.theta);
    ops::rope(k, k, pos_tensor, _meta.theta);
}

tensor_t Qwen2Model::_apply_attention(tensor_t q, tensor_t k, tensor_t v, size_t layer_idx, size_t seq_len, size_t start_pos) {
    using namespace llaisys;
    
    // 更新KV缓存
    auto k_cache_slice = _k_cache[layer_idx]->slice(0, start_pos, start_pos + seq_len);
    auto v_cache_slice = _v_cache[layer_idx]->slice(0, start_pos, start_pos + seq_len);
    
    // TODO: 实现tensor拷贝操作，或直接修改缓存
    // 当前简化版本：直接使用当前的K,V（没有真正使用缓存）
    
    // 计算注意力
    float scale = 1.0f / std::sqrt(static_cast<float>(_meta.dh));
    auto total_k = _k_cache[layer_idx]->slice(0, 0, start_pos + seq_len);
    auto total_v = _v_cache[layer_idx]->slice(0, 0, start_pos + seq_len);
    
    auto attn_out = _attn_output->slice(0, 0, seq_len);
    ops::self_attention(attn_out, q, total_k, total_v, scale);
    
    return attn_out;
}

int64_t Qwen2Model::_sample_token(tensor_t logits) {
    // 简单的argmax采样
    using namespace llaisys;
    
    auto max_idx_tensor = Tensor::create({1}, LLAISYS_DTYPE_I64, _device_type, _device_id);
    auto max_val_tensor = Tensor::create({1}, _meta.dtype, _device_type, _device_id);
    
    ops::argmax(max_idx_tensor, max_val_tensor, logits);
    
    // 读取结果
    std::vector<int64_t> result(1);
    // TODO: 实现从tensor读取数据的功能
    // 当前返回固定值作为占位符
    return _meta.end_token; // 临时返回结束token
}

} // namespace llaisys::models
