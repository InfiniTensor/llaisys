#include "llaisys/models/qwen2.h"
#include "../llaisys_tensor.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/add/op.hpp"

#include <vector>
#include <cmath>
#include <cstring>

using namespace llaisys;

// Qwen2模型结构
struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta;
    llaisysDeviceType_t device_type;
    int device_id;
    
    // 权重
    LlaisysQwen2Weights weights;
    
    // 中间张量（用于重用内存）
    tensor_t hidden_states;
    tensor_t residual;
    tensor_t q_proj;
    tensor_t k_proj;
    tensor_t v_proj;
    tensor_t o_proj;
    tensor_t q_rotated;
    tensor_t k_rotated;
    tensor_t attn_output;
    tensor_t gate_proj;
    tensor_t up_proj;
    tensor_t mlp_output;
    tensor_t logits;
    tensor_t max_val;
    tensor_t max_idx;
    
    // KV Cache
    std::vector<tensor_t> k_cache;
    std::vector<tensor_t> v_cache;
    
    // 位置ID张量
    tensor_t pos_ids;
};

// 辅助函数：从llaisysTensor_t获取tensor_t
inline tensor_t get_tensor(llaisysTensor_t t) {
    return t ? t->tensor : nullptr;
}

// 创建模型
__C struct LlaisysQwen2Model *llaisysQwen2ModelCreate(
    const LlaisysQwen2Meta *meta, 
    llaisysDeviceType_t device, 
    int *device_ids, 
    int ndevice) {
    
    auto *model = new LlaisysQwen2Model();
    model->meta = *meta;
    model->device_type = device;
    model->device_id = ndevice > 0 ? device_ids[0] : 0;
    
    size_t hs = meta->hs;
    size_t maxseq = meta->maxseq;
    
    // 创建中间张量
    model->hidden_states = Tensor::create({maxseq, hs}, meta->dtype, device, model->device_id);
    model->residual = Tensor::create({maxseq, hs}, meta->dtype, device, model->device_id);
    
    // QKV投影输出 [maxseq, nh * dh] 和 [maxseq, nkvh * dh]
    size_t q_size = meta->nh * meta->dh;
    size_t kv_size = meta->nkvh * meta->dh;
    model->q_proj = Tensor::create({maxseq, q_size}, meta->dtype, device, model->device_id);
    model->k_proj = Tensor::create({maxseq, kv_size}, meta->dtype, device, model->device_id);
    model->v_proj = Tensor::create({maxseq, kv_size}, meta->dtype, device, model->device_id);
    
    // 旋转后的QK [maxseq, nh, dh] 和 [maxseq, nkvh, dh]
    model->q_rotated = Tensor::create({maxseq, meta->nh, meta->dh}, meta->dtype, device, model->device_id);
    model->k_rotated = Tensor::create({maxseq, meta->nkvh, meta->dh}, meta->dtype, device, model->device_id);
    
    // 注意力输出 [maxseq, nh, dh]
    model->attn_output = Tensor::create({maxseq, meta->nh, meta->dh}, meta->dtype, device, model->device_id);
    
    // O投影输出 [maxseq, hs]
    model->o_proj = Tensor::create({maxseq, hs}, meta->dtype, device, model->device_id);
    
    // MLP中间张量
    model->gate_proj = Tensor::create({maxseq, meta->di}, meta->dtype, device, model->device_id);
    model->up_proj = Tensor::create({maxseq, meta->di}, meta->dtype, device, model->device_id);
    model->mlp_output = Tensor::create({maxseq, meta->di}, meta->dtype, device, model->device_id);
    
    // 输出logits [maxseq, voc]
    model->logits = Tensor::create({maxseq, meta->voc}, meta->dtype, device, model->device_id);
    
    // argmax输出
    model->max_val = Tensor::create({1}, meta->dtype, device, model->device_id);
    model->max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, device, model->device_id);
    
    // 创建权重张量
    // Embedding权重 [voc, hs]
    model->weights.in_embed = new LlaisysTensor{Tensor::create({meta->voc, hs}, meta->dtype, device, model->device_id)};
    model->weights.out_embed = new LlaisysTensor{Tensor::create({meta->voc, hs}, meta->dtype, device, model->device_id)};
    
    // 最终归一化权重 [hs]
    model->weights.out_norm_w = new LlaisysTensor{Tensor::create({hs}, meta->dtype, device, model->device_id)};
    
    // 分配每层权重指针数组
    model->weights.attn_norm_w = new llaisysTensor_t[meta->nlayer]();
    model->weights.attn_q_w = new llaisysTensor_t[meta->nlayer]();
    model->weights.attn_q_b = new llaisysTensor_t[meta->nlayer]();
    model->weights.attn_k_w = new llaisysTensor_t[meta->nlayer]();
    model->weights.attn_k_b = new llaisysTensor_t[meta->nlayer]();
    model->weights.attn_v_w = new llaisysTensor_t[meta->nlayer]();
    model->weights.attn_v_b = new llaisysTensor_t[meta->nlayer]();
    model->weights.attn_o_w = new llaisysTensor_t[meta->nlayer]();
    model->weights.mlp_norm_w = new llaisysTensor_t[meta->nlayer]();
    model->weights.mlp_gate_w = new llaisysTensor_t[meta->nlayer]();
    model->weights.mlp_up_w = new llaisysTensor_t[meta->nlayer]();
    model->weights.mlp_down_w = new llaisysTensor_t[meta->nlayer]();
    
    // 创建每层权重张量
    for (size_t i = 0; i < meta->nlayer; i++) {
        // Attention归一化权重 [hs]
        model->weights.attn_norm_w[i] = new LlaisysTensor{Tensor::create({hs}, meta->dtype, device, model->device_id)};
        
        // Q投影权重 [nh*dh, hs] 和偏置 [nh*dh]
        model->weights.attn_q_w[i] = new LlaisysTensor{Tensor::create({meta->nh * meta->dh, hs}, meta->dtype, device, model->device_id)};
        model->weights.attn_q_b[i] = new LlaisysTensor{Tensor::create({meta->nh * meta->dh}, meta->dtype, device, model->device_id)};
        
        // K投影权重 [nkvh*dh, hs] 和偏置 [nkvh*dh]
        model->weights.attn_k_w[i] = new LlaisysTensor{Tensor::create({meta->nkvh * meta->dh, hs}, meta->dtype, device, model->device_id)};
        model->weights.attn_k_b[i] = new LlaisysTensor{Tensor::create({meta->nkvh * meta->dh}, meta->dtype, device, model->device_id)};
        
        // V投影权重 [nkvh*dh, hs] 和偏置 [nkvh*dh]
        model->weights.attn_v_w[i] = new LlaisysTensor{Tensor::create({meta->nkvh * meta->dh, hs}, meta->dtype, device, model->device_id)};
        model->weights.attn_v_b[i] = new LlaisysTensor{Tensor::create({meta->nkvh * meta->dh}, meta->dtype, device, model->device_id)};
        
        // O投影权重 [hs, nh*dh]
        model->weights.attn_o_w[i] = new LlaisysTensor{Tensor::create({hs, meta->nh * meta->dh}, meta->dtype, device, model->device_id)};
        
        // MLP归一化权重 [hs]
        model->weights.mlp_norm_w[i] = new LlaisysTensor{Tensor::create({hs}, meta->dtype, device, model->device_id)};
        
        // Gate投影权重 [di, hs]
        model->weights.mlp_gate_w[i] = new LlaisysTensor{Tensor::create({meta->di, hs}, meta->dtype, device, model->device_id)};
        
        // Up投影权重 [di, hs]
        model->weights.mlp_up_w[i] = new LlaisysTensor{Tensor::create({meta->di, hs}, meta->dtype, device, model->device_id)};
        
        // Down投影权重 [hs, di]
        model->weights.mlp_down_w[i] = new LlaisysTensor{Tensor::create({hs, meta->di}, meta->dtype, device, model->device_id)};
    }
    
    // 初始化KV Cache
    model->k_cache.resize(meta->nlayer);
    model->v_cache.resize(meta->nlayer);
    for (size_t i = 0; i < meta->nlayer; i++) {
        model->k_cache[i] = Tensor::create({maxseq, meta->nkvh, meta->dh}, meta->dtype, device, model->device_id);
        model->v_cache[i] = Tensor::create({maxseq, meta->nkvh, meta->dh}, meta->dtype, device, model->device_id);
    }
    
    // 位置ID张量
    model->pos_ids = Tensor::create({maxseq}, LLAISYS_DTYPE_I64, device, model->device_id);
    
    return model;
}

// 销毁模型
__C void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
    if (!model) return;
    
    // 释放权重张量
    delete model->weights.in_embed;
    delete model->weights.out_embed;
    delete model->weights.out_norm_w;
    
    // 释放每层权重张量
    for (size_t i = 0; i < model->meta.nlayer; i++) {
        delete model->weights.attn_norm_w[i];
        delete model->weights.attn_q_w[i];
        delete model->weights.attn_q_b[i];
        delete model->weights.attn_k_w[i];
        delete model->weights.attn_k_b[i];
        delete model->weights.attn_v_w[i];
        delete model->weights.attn_v_b[i];
        delete model->weights.attn_o_w[i];
        delete model->weights.mlp_norm_w[i];
        delete model->weights.mlp_gate_w[i];
        delete model->weights.mlp_up_w[i];
        delete model->weights.mlp_down_w[i];
    }
    
    // 释放权重指针数组
    delete[] model->weights.attn_norm_w;
    delete[] model->weights.attn_q_w;
    delete[] model->weights.attn_q_b;
    delete[] model->weights.attn_k_w;
    delete[] model->weights.attn_k_b;
    delete[] model->weights.attn_v_w;
    delete[] model->weights.attn_v_b;
    delete[] model->weights.attn_o_w;
    delete[] model->weights.mlp_norm_w;
    delete[] model->weights.mlp_gate_w;
    delete[] model->weights.mlp_up_w;
    delete[] model->weights.mlp_down_w;
    
    delete model;
}

// 获取模型权重
__C struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
    return &model->weights;
}

// 辅助函数：复制张量数据到KV Cache
static void copy_to_kv_cache(tensor_t cache, tensor_t src, size_t start_pos, size_t len) {
    // src: [len, nkvh, dh]
    // cache: [maxseq, nkvh, dh]
    // 将src复制到cache的[start_pos, start_pos+len)位置
    
    auto cache_slice = cache->slice(0, start_pos, start_pos + len);
    
    // 获取数据指针
    std::byte *cache_data = cache_slice->data();
    const std::byte *src_data = src->data();
    
    size_t bytes = len * src->shape()[1] * src->shape()[2] * src->elementSize();
    std::memcpy(cache_data, src_data, bytes);
}

// 模型推理 - 单token
__C int64_t llaisysQwen2ModelInfer(
    struct LlaisysQwen2Model *model, 
    int64_t *token_ids, 
    size_t ntoken) {
    
    const auto &meta = model->meta;
    size_t hs = meta.hs;
    size_t nh = meta.nh;
    size_t nkvh = meta.nkvh;
    size_t dh = meta.dh;
    size_t di = meta.di;
    size_t voc = meta.voc;
    size_t nlayer = meta.nlayer;
    float epsilon = meta.epsilon;
    float theta = meta.theta;
    
    // 使用hidden_states的前ntoken行
    auto hidden = model->hidden_states->slice(0, 0, ntoken);
    
    // 1. Embedding
    auto token_ids_tensor = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, model->device_type, model->device_id);
    token_ids_tensor->load(token_ids);
    ops::embedding(hidden, token_ids_tensor, get_tensor(model->weights.in_embed));
    
    // 设置位置ID
    auto pos_ids_slice = model->pos_ids->slice(0, 0, ntoken);
    std::vector<int64_t> pos_ids_data(ntoken);
    for (size_t i = 0; i < ntoken; i++) {
        pos_ids_data[i] = static_cast<int64_t>(i);
    }
    pos_ids_slice->load(pos_ids_data.data());
    
    // Transformer层
    for (size_t layer = 0; layer < nlayer; layer++) {
        // 保存残差
        auto residual = hidden;
        
        // 1. RMS Norm (input_layernorm)
        ops::rms_norm(hidden, hidden, get_tensor(model->weights.attn_norm_w[layer]), epsilon);
        
        // 2. QKV投影
        auto q_proj_slice = model->q_proj->slice(0, 0, ntoken);
        auto k_proj_slice = model->k_proj->slice(0, 0, ntoken);
        auto v_proj_slice = model->v_proj->slice(0, 0, ntoken);
        
        ops::linear(q_proj_slice, hidden, get_tensor(model->weights.attn_q_w[layer]), 
                    get_tensor(model->weights.attn_q_b[layer]));
        ops::linear(k_proj_slice, hidden, get_tensor(model->weights.attn_k_w[layer]),
                    get_tensor(model->weights.attn_k_b[layer]));
        ops::linear(v_proj_slice, hidden, get_tensor(model->weights.attn_v_w[layer]),
                    get_tensor(model->weights.attn_v_b[layer]));
        
        // 3. 重塑为 [ntoken, nh, dh] 和 [ntoken, nkvh, dh]
        auto q_reshaped = q_proj_slice->view({ntoken, nh, dh});
        auto k_reshaped = k_proj_slice->view({ntoken, nkvh, dh});
        auto v_reshaped = v_proj_slice->view({ntoken, nkvh, dh});
        
        // 4. RoPE
        auto q_rotated_slice = model->q_rotated->slice(0, 0, ntoken);
        auto k_rotated_slice = model->k_rotated->slice(0, 0, ntoken);
        
        ops::rope(q_rotated_slice, q_reshaped, pos_ids_slice, theta);
        ops::rope(k_rotated_slice, k_reshaped, pos_ids_slice, theta);
        
        // 5. 更新KV Cache
        copy_to_kv_cache(model->k_cache[layer], k_rotated_slice, 0, ntoken);
        copy_to_kv_cache(model->v_cache[layer], v_reshaped, 0, ntoken);
        
        // 6. Self Attention
        auto k_cache_slice = model->k_cache[layer]->slice(0, 0, ntoken);
        auto v_cache_slice = model->v_cache[layer]->slice(0, 0, ntoken);
        
        auto attn_output_slice = model->attn_output->slice(0, 0, ntoken);
        float scale = 1.0f / std::sqrt(static_cast<float>(dh));
        ops::self_attention(attn_output_slice, q_rotated_slice, k_cache_slice, v_cache_slice, scale);
        
        // 7. O投影
        auto o_proj_slice = model->o_proj->slice(0, 0, ntoken);
        auto attn_flat = attn_output_slice->view({ntoken, nh * dh});
        ops::linear(o_proj_slice, attn_flat, get_tensor(model->weights.attn_o_w[layer]), nullptr);
        
        // 8. 残差连接
        ops::add(o_proj_slice, o_proj_slice, residual);
        
        // 9. 保存残差用于MLP
        residual = o_proj_slice;
        
        // 10. RMS Norm (post_attention_layernorm)
        ops::rms_norm(o_proj_slice, o_proj_slice, get_tensor(model->weights.mlp_norm_w[layer]), epsilon);
        
        // 11. MLP
        auto gate_slice = model->gate_proj->slice(0, 0, ntoken);
        auto up_slice = model->up_proj->slice(0, 0, ntoken);
        
        ops::linear(gate_slice, o_proj_slice, get_tensor(model->weights.mlp_gate_w[layer]), nullptr);
        ops::linear(up_slice, o_proj_slice, get_tensor(model->weights.mlp_up_w[layer]), nullptr);
        
        auto mlp_out_slice = model->mlp_output->slice(0, 0, ntoken);
        ops::swiglu(mlp_out_slice, gate_slice, up_slice);
        
        // 12. Down投影
        ops::linear(hidden, mlp_out_slice, get_tensor(model->weights.mlp_down_w[layer]), nullptr);
        
        // 13. 残差连接
        ops::add(hidden, hidden, residual);
    }
    
    // 最终RMS Norm
    ops::rms_norm(hidden, hidden, get_tensor(model->weights.out_norm_w), epsilon);
    
    // 输出投影
    auto logits_slice = model->logits->slice(0, 0, ntoken);
    ops::linear(logits_slice, hidden, get_tensor(model->weights.out_embed), nullptr);
    
    // 取最后一个token的logits进行argmax
    auto last_logits = logits_slice->slice(0, ntoken - 1, ntoken);
    auto last_logits_flat = last_logits->view({voc});
    
    ops::argmax(model->max_idx, model->max_val, last_logits_flat);
    
    // 获取结果
    int64_t result;
    std::memcpy(&result, model->max_idx->data(), sizeof(int64_t));
    
    return result;
}
