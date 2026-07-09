#include "qwen2_model.hpp"
#include "../../llaisys/llaisys_tensor.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/add/op.hpp"
#include "../../ops/rearrange/op.hpp"
#include "../../utils.hpp"
#include <cmath>
#include <cstring>

namespace llaisys::models {

Qwen2Model::Qwen2Model(const LlaisysQwen2Meta *meta_, llaisysDeviceType_t device, int dev_id)
    : device_type(device), device_id(dev_id), cur_seq_len(0) {
    meta = *meta_;
    
    in_embed = Tensor::create({meta.voc, meta.hs}, meta.dtype, device, dev_id);
    out_embed = Tensor::create({meta.voc, meta.hs}, meta.dtype, device, dev_id);
    out_norm_w = Tensor::create({meta.hs}, meta.dtype, device, dev_id);
    
    for (size_t i = 0; i < meta.nlayer; i++) {
        attn_norm_w.push_back(Tensor::create({meta.hs}, meta.dtype, device, dev_id));
        attn_q_w.push_back(Tensor::create({meta.nh * meta.dh, meta.hs}, meta.dtype, device, dev_id));
        attn_q_b.push_back(Tensor::create({meta.nh * meta.dh}, meta.dtype, device, dev_id));
        attn_k_w.push_back(Tensor::create({meta.nkvh * meta.dh, meta.hs}, meta.dtype, device, dev_id));
        attn_k_b.push_back(Tensor::create({meta.nkvh * meta.dh}, meta.dtype, device, dev_id));
        attn_v_w.push_back(Tensor::create({meta.nkvh * meta.dh, meta.hs}, meta.dtype, device, dev_id));
        attn_v_b.push_back(Tensor::create({meta.nkvh * meta.dh}, meta.dtype, device, dev_id));
        attn_o_w.push_back(Tensor::create({meta.hs, meta.nh * meta.dh}, meta.dtype, device, dev_id));
        
        mlp_norm_w.push_back(Tensor::create({meta.hs}, meta.dtype, device, dev_id));
        mlp_gate_w.push_back(Tensor::create({meta.di, meta.hs}, meta.dtype, device, dev_id));
        mlp_up_w.push_back(Tensor::create({meta.di, meta.hs}, meta.dtype, device, dev_id));
        mlp_down_w.push_back(Tensor::create({meta.hs, meta.di}, meta.dtype, device, dev_id));
        
        k_cache.push_back(Tensor::create({meta.maxseq, meta.nkvh, meta.dh}, meta.dtype, device, dev_id));
        v_cache.push_back(Tensor::create({meta.maxseq, meta.nkvh, meta.dh}, meta.dtype, device, dev_id));
    }
}

LlaisysQwen2Weights Qwen2Model::getWeights() {
    LlaisysQwen2Weights weights;
    weights.in_embed = new LlaisysTensor{in_embed};
    weights.out_embed = new LlaisysTensor{out_embed};
    weights.out_norm_w = new LlaisysTensor{out_norm_w};
    
    weights.attn_norm_w = new llaisysTensor_t[meta.nlayer];
    weights.attn_q_w = new llaisysTensor_t[meta.nlayer];
    weights.attn_q_b = new llaisysTensor_t[meta.nlayer];
    weights.attn_k_w = new llaisysTensor_t[meta.nlayer];
    weights.attn_k_b = new llaisysTensor_t[meta.nlayer];
    weights.attn_v_w = new llaisysTensor_t[meta.nlayer];
    weights.attn_v_b = new llaisysTensor_t[meta.nlayer];
    weights.attn_o_w = new llaisysTensor_t[meta.nlayer];
    weights.mlp_norm_w = new llaisysTensor_t[meta.nlayer];
    weights.mlp_gate_w = new llaisysTensor_t[meta.nlayer];
    weights.mlp_up_w = new llaisysTensor_t[meta.nlayer];
    weights.mlp_down_w = new llaisysTensor_t[meta.nlayer];
    
    for (size_t i = 0; i < meta.nlayer; i++) {
        weights.attn_norm_w[i] = new LlaisysTensor{attn_norm_w[i]};
        weights.attn_q_w[i] = new LlaisysTensor{attn_q_w[i]};
        weights.attn_q_b[i] = new LlaisysTensor{attn_q_b[i]};
        weights.attn_k_w[i] = new LlaisysTensor{attn_k_w[i]};
        weights.attn_k_b[i] = new LlaisysTensor{attn_k_b[i]};
        weights.attn_v_w[i] = new LlaisysTensor{attn_v_w[i]};
        weights.attn_v_b[i] = new LlaisysTensor{attn_v_b[i]};
        weights.attn_o_w[i] = new LlaisysTensor{attn_o_w[i]};
        weights.mlp_norm_w[i] = new LlaisysTensor{mlp_norm_w[i]};
        weights.mlp_gate_w[i] = new LlaisysTensor{mlp_gate_w[i]};
        weights.mlp_up_w[i] = new LlaisysTensor{mlp_up_w[i]};
        weights.mlp_down_w[i] = new LlaisysTensor{mlp_down_w[i]};
    }
    
    return weights;
}

int64_t Qwen2Model::infer(int64_t *token_ids, size_t ntoken) {
    //向前传播
    size_t seqlen = ntoken - cur_seq_len;
    
    //embedding
    auto idx_tensor = Tensor::create({seqlen}, LLAISYS_DTYPE_I64, device_type, device_id);
    idx_tensor->load(token_ids + cur_seq_len);
    auto x = Tensor::create({seqlen, meta.hs}, meta.dtype, device_type, device_id);
    ops::embedding(x, idx_tensor, in_embed);
    
    //位置编码ID
    std::vector<int64_t> pos_ids_vec(seqlen);
    for (size_t i = 0; i < seqlen; i++) pos_ids_vec[i] = cur_seq_len + i;
    auto pos_ids = Tensor::create({seqlen}, LLAISYS_DTYPE_I64, device_type, device_id);
    pos_ids->load(pos_ids_vec.data());
    
    //Transformer层
    for (size_t layer = 0; layer < meta.nlayer; layer++) {
        
        auto x_norm = Tensor::create({seqlen, meta.hs}, meta.dtype, device_type, device_id);
        ops::rms_norm(x_norm, x, attn_norm_w[layer], meta.epsilon);
        
        auto q = Tensor::create({seqlen, meta.nh * meta.dh}, meta.dtype, device_type, device_id);
        auto k = Tensor::create({seqlen, meta.nkvh * meta.dh}, meta.dtype, device_type, device_id);
        auto v = Tensor::create({seqlen, meta.nkvh * meta.dh}, meta.dtype, device_type, device_id);
        ops::linear(q, x_norm, attn_q_w[layer], attn_q_b[layer]);
        ops::linear(k, x_norm, attn_k_w[layer], attn_k_b[layer]);
        ops::linear(v, x_norm, attn_v_w[layer], attn_v_b[layer]);
        
        //重塑
        q = q->view({seqlen, meta.nh, meta.dh});
        k = k->view({seqlen, meta.nkvh, meta.dh});
        v = v->view({seqlen, meta.nkvh, meta.dh});
        
        //rope
        auto q_rope = Tensor::create({seqlen, meta.nh, meta.dh}, meta.dtype, device_type, device_id);
        auto k_rope = Tensor::create({seqlen, meta.nkvh, meta.dh}, meta.dtype, device_type, device_id);
        ops::rope(q_rope, q, pos_ids, meta.theta);
        ops::rope(k_rope, k, pos_ids, meta.theta);
        
        //更新KV cache
        auto k_cache_slice = k_cache[layer]->slice(0, cur_seq_len, cur_seq_len + seqlen);
        auto v_cache_slice = v_cache[layer]->slice(0, cur_seq_len, cur_seq_len + seqlen);
        ops::rearrange(k_cache_slice, k_rope);
        ops::rearrange(v_cache_slice, v);
        
        auto k_full = k_cache[layer]->slice(0, 0, cur_seq_len + seqlen);
        auto v_full = v_cache[layer]->slice(0, 0, cur_seq_len + seqlen);
        
        //self attention
        auto attn_out = Tensor::create({seqlen, meta.nh, meta.dh}, meta.dtype, device_type, device_id);
        float scale = 1.0f / std::sqrt(static_cast<float>(meta.dh));
        ops::self_attention(attn_out, q_rope, k_full, v_full, scale);
        
        attn_out = attn_out->view({seqlen, meta.nh * meta.dh});
        auto attn_proj = Tensor::create({seqlen, meta.hs}, meta.dtype, device_type, device_id);
        ops::linear(attn_proj, attn_out, attn_o_w[layer], nullptr);
        
        ops::add(x, x, attn_proj);
        
        //MLP
        auto x_mlp = Tensor::create({seqlen, meta.hs}, meta.dtype, device_type, device_id);
        ops::rms_norm(x_mlp, x, mlp_norm_w[layer], meta.epsilon);
        
        auto gate = Tensor::create({seqlen, meta.di}, meta.dtype, device_type, device_id);
        auto up = Tensor::create({seqlen, meta.di}, meta.dtype, device_type, device_id);
        ops::linear(gate, x_mlp, mlp_gate_w[layer], nullptr);
        ops::linear(up, x_mlp, mlp_up_w[layer], nullptr);
        
        auto mlp_out = Tensor::create({seqlen, meta.di}, meta.dtype, device_type, device_id);
        ops::swiglu(mlp_out, gate, up);
        
        auto mlp_proj = Tensor::create({seqlen, meta.hs}, meta.dtype, device_type, device_id);
        ops::linear(mlp_proj, mlp_out, mlp_down_w[layer], nullptr);
        
        // residual
        ops::add(x, x, mlp_proj);
    }
    
    //归一化
    auto x_final = Tensor::create({seqlen, meta.hs}, meta.dtype, device_type, device_id);
    ops::rms_norm(x_final, x, out_norm_w, meta.epsilon);
    
    //用最后一个预测
    auto last_hidden = x_final->slice(0, seqlen - 1, seqlen);
    auto logits = Tensor::create({1, meta.voc}, meta.dtype, device_type, device_id);
    ops::linear(logits, last_hidden, out_embed, nullptr);
    
    //argmax
    auto max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, device_type, device_id);
    auto max_val = Tensor::create({1}, meta.dtype, device_type, device_id);
    ops::argmax(max_idx, max_val, logits->view({meta.voc}));
    
    int64_t result;
    std::byte *data = max_idx->data();
    std::memcpy(&result, data, sizeof(int64_t));
    
    cur_seq_len += seqlen;
    return result;
}

}
