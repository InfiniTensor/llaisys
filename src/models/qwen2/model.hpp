#pragma once

#include "../../tensor/tensor.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../ops/argmax/op.hpp"

#include <vector>
#include <memory>

namespace llaisys::models::qwen2 {
// 模型元数据
    struct ModelMeta {
    llaisysDataType_t dtype;
    size_t nlayer;      // 层数
    size_t hs;          // hidden size
    size_t nh;          // num heads
    size_t nkvh;        // num kv heads
    size_t dh;          // head dimension
    size_t di;          // intermediate size
    size_t maxseq;      // max sequence length
    size_t voc;         // vocabulary size
    float epsilon;      // RMS norm epsilon
    float theta;        // RoPE theta
    int64_t end_token;  // end token id
};

// 模型权重
    struct ModelWeights {
    tensor_t in_embed;           // [voc, hs]
    tensor_t out_embed;          // [voc, hs]
    tensor_t out_norm_w;         // [hs]
    
    // 每层的权重
    std::vector<tensor_t> attn_norm_w;  // [nlayer] x [hs]
    std::vector<tensor_t> attn_q_w;     // [nlayer] x [nh * dh, hs]
    std::vector<tensor_t> attn_q_b;     // [nlayer] x [nh * dh] (可能为空)
    std::vector<tensor_t> attn_k_w;     // [nlayer] x [nkvh * dh, hs]
    std::vector<tensor_t> attn_k_b;     // [nlayer] x [nkvh * dh] (可能为空)
    std::vector<tensor_t> attn_v_w;     // [nlayer] x [nkvh * dh, hs]
    std::vector<tensor_t> attn_v_b;     // [nlayer] x [nkvh * dh] (可能为空)
    std::vector<tensor_t> attn_o_w;      // [nlayer] x [hs, nh * dh]
    
    std::vector<tensor_t> mlp_norm_w;   // [nlayer] x [hs]
    std::vector<tensor_t> mlp_gate_w;    // [nlayer] x [di, hs]
    std::vector<tensor_t> mlp_up_w;      // [nlayer] x [di, hs]
    std::vector<tensor_t> mlp_down_w;   // [nlayer] x [hs, di]
};

// 模型类
class Model {
private:
    ModelMeta meta_;
    ModelWeights weights_;
    llaisysDeviceType_t device_type_;
    int device_id_;
    
    // KV Cache: 每层的 K 和 V
    std::vector<tensor_t> k_cache_;  // [nlayer] x [maxseq, nkvh, dh]
    std::vector<tensor_t> v_cache_;  // [nlayer] x [maxseq, nkvh, dh]
    size_t cache_len_;               // 当前 cache 长度
    
    // Dummy bias tensors（用于没有 bias 的层，必须全零）
    tensor_t dummy_bias_hs_;         // [hs] - 用于 o_proj, mlp_down, out_embed
    tensor_t dummy_bias_di_;         // [di] - 用于 mlp_gate, mlp_up
    tensor_t dummy_bias_q_;          // [nh * dh] - 用于 q_proj
    tensor_t dummy_bias_kv_;         // [nkvh * dh] - 用于 k_proj, v_proj
    tensor_t dummy_bias_voc_;         // [voc] - 用于 out_embed
    
    // 临时张量（避免重复分配）
    tensor_t x_;                     // 当前隐藏状态 [seqlen, hs]
    tensor_t x_norm_;                // 归一化后的隐藏状态
    tensor_t q_flat_;                // [seqlen, nh * dh]
    tensor_t k_flat_;                // [seqlen, nkvh * dh]
    tensor_t v_flat_;                // [seqlen, nkvh * dh]
    tensor_t q_;                     // Query [seqlen, nh, dh]
    tensor_t k_;                     // Key [seqlen, nkvh, dh]
    tensor_t v_;                     // Value [seqlen, nkvh, dh]
    tensor_t q_rope_;                // [seqlen, nh, dh]
    tensor_t k_rope_new_;            // [seqlen, nkvh, dh]
    tensor_t k_full_;                // 完整的 K（包含 cache）[total_len, nkvh, dh]
    tensor_t v_full_;                // 完整的 V（包含 cache）[total_len, nkvh, dh]
    tensor_t attn_out_;              // Attention 输出 [seqlen, nh, dh]
    tensor_t attn_proj_out_;          // Attention 投影输出 [seqlen, hs]
    tensor_t x_attn_;                // Attention 残差输出 [seqlen, hs]
    tensor_t gate_;                   // MLP gate [seqlen, di]
    tensor_t up_;                     // MLP up [seqlen, di]
    tensor_t swiglu_out_;            // SwiGLU 输出 [seqlen, di]
    tensor_t mlp_out_;                // MLP 输出 [seqlen, hs]
    tensor_t x_mlp_;                 // MLP 残差输出 [seqlen, hs]
    tensor_t logits_;                // 输出 logits [seqlen, voc]
    tensor_t pos_ids_q_;             // 位置 id [seqlen]
    tensor_t input_ids_buf_;         // infer 输入缓存 [ntoken]
    tensor_t max_idx_;               // argmax 索引缓存 [1]
    tensor_t max_val_;               // argmax 值缓存 [1]
    
    // 前向传播辅助函数
    void ensure_tensor(tensor_t &tensor, const std::vector<size_t> &shape, llaisysDataType_t dtype);
    void forward_layer(size_t layer_idx, tensor_t& x, size_t seqlen, size_t total_len, tensor_t pos_ids_q);
    void update_kv_cache(size_t layer_idx, tensor_t k_new, tensor_t v_new, size_t seqlen, size_t old_len);
    
public:
    Model(const ModelMeta& meta, llaisysDeviceType_t device_type, int device_id);
    ~Model();
    
    ModelWeights& weights() { return weights_; }
    const ModelWeights& weights() const { return weights_; }
    const ModelMeta& meta() const { return meta_; }
    
    // 前向传播：返回 logits
    tensor_t forward(tensor_t input_ids, size_t seqlen, size_t total_len);
    
    // 推理：生成下一个 token
    int64_t infer(
        int64_t* token_ids,
        size_t ntoken,
        int top_k = 1,
        float top_p = 1.0f,
        float temperature = 1.0f);
    
    // 重置 KV Cache
    void reset_cache();
};

} // namespace llaisys::models::qwen2
