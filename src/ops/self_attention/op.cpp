#include "op.hpp"
#include <cmath>


namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    //TO_BE_IMPLEMENTED();
    // 获取张量数据指针
    auto attn_val_data = attn_val->data();
    auto q_data = q->data();
    auto k_data = k->data();
    auto v_data = v->data();
    
    // 获取维度信息
    auto q_shape = q->shape();
    auto k_shape = k->shape();
    auto v_shape = v->shape();

    const int64_t seqlen    = (int64_t)q_shape[0];   // L
    const int64_t nhead     = (int64_t)q_shape[1];   // nh
    const int64_t d         = (int64_t)q_shape[2];   // hd
    const int64_t total_len = (int64_t)k_shape[0];   // S
    const int64_t nkvhead   = (int64_t)k_shape[1];   // nkvh
    const int64_t dv        = (int64_t)v_shape[2];



    // Must match torch:
    // key/value = repeat_interleave(head_group) along head dim => kv_head_idx = h / head_group
    // Causal mask uses tril(diagonal=S-L) => allow tk <= sq + (S-L)

    const int64_t diag = total_len - seqlen;         // diagonal = S - L

    // repeat_interleave(head_group) 的语义：kv_idx = h / head_group
    // 测试用例保证整除，否则这里需要处理
    const int64_t head_group = (nkvhead == 0) ? 1 : (nhead / nkvhead);
    
    auto dtype = q->dtype();
    
    // 根据数据类型进行处理
    switch (static_cast<llaisysDataType_t>(dtype)) {
    case LLAISYS_DTYPE_F32: {
        auto* q_ptr = reinterpret_cast<const float*>(q_data);
        auto* k_ptr = reinterpret_cast<const float*>(k_data);
        auto* v_ptr = reinterpret_cast<const float*>(v_data);
        //auto* attn_val_ptr = reinterpret_cast<float*>(attn_val_data);
        auto* out   = reinterpret_cast<float*>(attn_val_data);
        
        // 为每个查询位置计算注意力
        //for (size_t sq = 0; sq < seqlen; sq++) {
        for (int64_t sq = 0; sq < seqlen; ++sq) {
            // torch: tril(diagonal=S-L) => allow tk <= sq + (S-L)
            int64_t max_k = sq + diag;
            int64_t valid_len_i64 = max_k + 1;
            if (valid_len_i64 < 0) valid_len_i64 = 0;
            if (valid_len_i64 > total_len) valid_len_i64 = total_len;
            const int64_t valid_len = valid_len_i64;
            //for (size_t h = 0; h < nhead; h++) {
            for (int64_t h = 0; h < nhead; ++h) {

                // size_t kv_head_idx = h / head_group;
                // // 计算对应的 kv 头索引（注意：这是 repeat_interleave，不是分组共享）
                // // 在 repeat_interleave 中，KV头被重复了 head_group 次
                // // 所以索引应该是：h % nkvhead
                // //size_t kv_head_idx = h % nkvhead;
                const int64_t kv_head_idx = (head_group > 0) ? (h / head_group) : 0;



                if (seqlen == 5 && total_len == 11 && sq == 0 && h == 0) {
    std::cout << "[DBG] nhead=" << nhead
              << " nkvhead=" << nkvhead
              << " head_group=" << head_group
              << " diag=" << diag
              << std::endl;
}
if (seqlen == 5 && total_len == 11 && sq == 0) {
    std::cout << "[DBG] h=" << h
              << " kv_head_idx=" << kv_head_idx
              << std::endl;
}

if (seqlen == 5 && total_len == 11 && sq == 0 && (h == 0 || h == 2)) {
    std::cout << "[DBG] h=" << h << " kv=" << kv_head_idx << std::endl;
}



                
                const float* q_head = q_ptr + (sq * nhead + h) * d;
                //float* attn_head = attn_val_ptr + (sq * nhead + h) * dv;
                float* out_head      = out   + (sq * nhead + h) * dv;


                // init output
                for (int64_t i = 0; i < dv; ++i) out_head[i] = 0.0f;

                if (valid_len == 0) {
                    // 与 torch 在极端形状下的行为可能不同（全 -inf softmax -> NaN）
                    // 测试用例不会触发；这里先直接返回 0
                    continue;
                }
                
                // scores buffer (float)
                float* scores = new float[(size_t)valid_len];


                // // 初始化注意力值为0
                // for (size_t dv_idx = 0; dv_idx < dv; dv_idx++) {
                //     attn_head[dv_idx] = 0.0f;
                // }
                
                // 计算注意力分数
                float max_score = -INFINITY;
                // // 只考虑当前位置及之前的位置（因果注意力）
                // // 注意：sq是查询位置，tk是键位置
                // //size_t valid_len = (sq < total_len) ? sq + 1 : total_len;
                // size_t valid_len = std::min(sq + 1, total_len);
                // //float* attn_scores = new float[total_len];
                // float* attn_scores = new float[valid_len];
                for (int64_t tk = 0; tk < valid_len; ++tk) {
                    const float* k_head = k_ptr + (tk * nkvhead + kv_head_idx) * d;

                    float dot = 0.0f;
                    for (int64_t i = 0; i < d; ++i) dot += q_head[i] * k_head[i];

                    float s = dot * scale;
                    scores[tk] = s;
                    if (s > max_score) max_score = s;
                }
                
                // // 计算 Q·K^T：查询头与对应KV头的点积
                // //for (size_t tk = 0; tk <= sq; tk++) {
                // for (size_t tk = 0; tk < valid_len; tk++) {
                //      const float* k_head = k_ptr + (tk * nkvhead + kv_head_idx) * d;
                    
                //     float score = 0.0f;
                //     for (size_t idx = 0; idx < d; idx++) {
                //         score += q_head[idx] * k_head[idx];
                //     }
                //     score *= scale;

                //     // 应用因果掩码：未来位置设为负无穷
                //     // 由于我们只循环到 valid_len = min(sq+1, total_len)
                //     // 所以不需要额外的掩码判断
                    
                //     // if (tk > sq) {
                //     //     score = -INFINITY;
                //     //     //score = -10000.0;//-INFINITY;
                //     // }
                //     // 存储分数
                //     attn_scores[tk] = score;
                //     if (score > max_score) {
                //         max_score = score;
                //     }
                // }
                
                // // 计算 softmax（数值稳定版本）
                // float exp_sum = 0.0f;
                // //for (size_t tk = 0; tk <= sq; tk++) {
                // for (size_t tk = 0; tk < valid_len; tk++) {
                //     // 对于因果注意力，未来位置已经在valid_len中排除了
                //     // 所以这里直接计算softmax
                //     float exp_val = std::exp(attn_scores[tk] - max_score);
                //     exp_sum += exp_val;
                //     attn_scores[tk] = exp_val;
                // }
                
                // // 归一化并加权求和
                // //for (size_t tk = 0; tk <= sq; tk++) {
                // for (size_t tk = 0; tk < valid_len; tk++) {
                //     float weight = attn_scores[tk] / exp_sum;
                //     const float* v_head = v_ptr + (tk * nkvhead + kv_head_idx) * dv;
                    
                //     for (size_t dv_idx = 0; dv_idx < dv; dv_idx++) {
                //         attn_head[dv_idx] += weight * v_head[dv_idx];
                //     }
                // }

                                float exp_sum = 0.0f;
                for (int64_t tk = 0; tk < valid_len; ++tk) {
                    float e = std::exp(scores[tk] - max_score);
                    scores[tk] = e;
                    exp_sum += e;
                }

                const float inv_sum = 1.0f / exp_sum;
                for (int64_t tk = 0; tk < valid_len; ++tk) {
                    const float w = scores[tk] * inv_sum;
                    const float* v_head = v_ptr + (tk * nkvhead + kv_head_idx) * dv;
                    for (int64_t i = 0; i < dv; ++i) out_head[i] += w * v_head[i];
                }

                
                //delete[] attn_scores;
                delete[] scores;
            }
        }
        break;
    }
    
    case LLAISYS_DTYPE_F16: {
        auto* q_ptr = reinterpret_cast<const llaisys::fp16_t*>(q_data);
        auto* k_ptr = reinterpret_cast<const llaisys::fp16_t*>(k_data);
        auto* v_ptr = reinterpret_cast<const llaisys::fp16_t*>(v_data);
        //auto* attn_val_ptr = reinterpret_cast<llaisys::fp16_t*>(attn_val_data);
        auto* out   = reinterpret_cast<llaisys::fp16_t*>(attn_val_data);
        
        //for (size_t sq = 0; sq < seqlen; sq++) {
        for (int64_t sq = 0; sq < seqlen; ++sq) {
            int64_t max_k = sq + diag;
            int64_t valid_len_i64 = max_k + 1;
            if (valid_len_i64 < 0) valid_len_i64 = 0;
            if (valid_len_i64 > total_len) valid_len_i64 = total_len;
            const int64_t valid_len = valid_len_i64;
            
            
            
            
            //for (size_t h = 0; h < nhead; h++) {
            for (int64_t h = 0; h < nhead; ++h) {
                // //size_t kv_head_idx = h / head_group;
                // // 使用取模运算，匹配 repeat_interleave
                // size_t kv_head_idx = h % nkvhead;
                const int64_t kv_head_idx = (head_group > 0) ? (h / head_group) : 0;
                
                const llaisys::fp16_t* q_head = q_ptr + (sq * nhead + h) * d;
                //llaisys::fp16_t* attn_head = attn_val_ptr + (sq * nhead + h) * dv;
                llaisys::fp16_t* out_head     = out   + (sq * nhead + h) * dv;
                
                // // 初始化注意力值为0
                // for (size_t dv_idx = 0; dv_idx < dv; dv_idx++) {
                //     //attn_head[dv_idx] = llaisys::fp16_t(0.0f);
                //     attn_head[dv_idx] = llaisys::utils::_f32_to_f16(0.0f);
                // }

                // float accumulator to reduce quantization error
                float* acc = new float[(size_t)dv];
                for (int64_t i = 0; i < dv; ++i) acc[i] = 0.0f;

                if (valid_len == 0) {
                    for (int64_t i = 0; i < dv; ++i) out_head[i] = llaisys::utils::_f32_to_f16(0.0f);
                    delete[] acc;
                    continue;
                }

                float* scores = new float[(size_t)valid_len];


                
                float max_score = -INFINITY;
                // //size_t valid_len = (sq < total_len) ? sq + 1 : total_len;
                // size_t valid_len = std::min(sq + 1, total_len);
                
                // //float* attn_scores = new float[total_len];
                // float* attn_scores = new float[valid_len];
                
                // // 计算 Q·K^T
                // //for (size_t tk = 0; tk <= sq; tk++) {
                // for (size_t tk = 0; tk < valid_len; tk++) {
                //     const llaisys::fp16_t* k_head = k_ptr + (tk * nkvhead + kv_head_idx) * d;
                    
                //     float score = 0.0f;
                //     for (size_t idx = 0; idx < d; idx++) {
                //         float q_val = llaisys::utils::_f16_to_f32(q_head[idx]);
                //         float k_val = llaisys::utils::_f16_to_f32(k_head[idx]);
                //         score += q_val * k_val;
                //     }
                //     score *= scale;
                    
                //     // if (tk > sq) {
                //     //     score = -INFINITY;
                //     // }
                    
                //      // 存储分数
                //     attn_scores[tk] = score;
                //     if (score > max_score) {
                //         max_score = score;
                //     }
                // }
                
                // // 计算 softmax
                // float exp_sum = 0.0f;
                // //for (size_t tk = 0; tk <= sq; tk++) {
                // for (size_t tk = 0; tk < valid_len; tk++) {
                //     // 对于因果注意力，未来位置已经在valid_len中排除了
                //     // 所以这里直接计算softmax
                //     float exp_val = std::exp(attn_scores[tk] - max_score);
                //     exp_sum += exp_val;
                //     attn_scores[tk] = exp_val;
                // }
                
                // // 加权求和
                // //for (size_t tk = 0; tk <= sq; tk++) {
                // for (size_t tk = 0; tk < valid_len; tk++) {
                //     float weight = attn_scores[tk] / exp_sum;
                //     const llaisys::fp16_t* v_head = v_ptr + (tk * nkvhead + kv_head_idx) * dv;
                    
                //     for (size_t dv_idx = 0; dv_idx < dv; dv_idx++) {
                //         float attn_val_f = llaisys::utils::_f16_to_f32(attn_head[dv_idx]);
                //         float v_val = llaisys::utils::_f16_to_f32(v_head[dv_idx]);
                //         attn_val_f += weight * v_val;
                //         attn_head[dv_idx] = llaisys::utils::_f32_to_f16(attn_val_f);
                //     }
                // }
                
                // delete[] attn_scores;
                max_score = -INFINITY;
                for (int64_t tk = 0; tk < valid_len; ++tk) {
                    const llaisys::fp16_t* k_head = k_ptr + (tk * nkvhead + kv_head_idx) * d;

                    float dot = 0.0f;
                    for (int64_t i = 0; i < d; ++i) {
                        float qv = llaisys::utils::_f16_to_f32(q_head[i]);
                        float kv = llaisys::utils::_f16_to_f32(k_head[i]);
                        dot += qv * kv;
                    }

                    float s = dot * scale;
                    scores[tk] = s;
                    if (s > max_score) max_score = s;
                }

                float exp_sum = 0.0f;
                for (int64_t tk = 0; tk < valid_len; ++tk) {
                    float e = std::exp(scores[tk] - max_score);
                    scores[tk] = e;
                    exp_sum += e;
                }

                const float inv_sum = 1.0f / exp_sum;
                for (int64_t tk = 0; tk < valid_len; ++tk) {
                    const float w = scores[tk] * inv_sum;
                    const llaisys::fp16_t* v_head = v_ptr + (tk * nkvhead + kv_head_idx) * dv;
                    for (int64_t i = 0; i < dv; ++i) {
                        float vv = llaisys::utils::_f16_to_f32(v_head[i]);
                        acc[i] += w * vv;
                    }
                }

                for (int64_t i = 0; i < dv; ++i) out_head[i] = llaisys::utils::_f32_to_f16(acc[i]);

                delete[] scores;
                delete[] acc;
            }
        }
        break;
    }
    
    case LLAISYS_DTYPE_BF16: {
        auto* q_ptr = reinterpret_cast<const llaisys::bf16_t*>(q_data);
        auto* k_ptr = reinterpret_cast<const llaisys::bf16_t*>(k_data);
        auto* v_ptr = reinterpret_cast<const llaisys::bf16_t*>(v_data);
        //auto* attn_val_ptr = reinterpret_cast<llaisys::bf16_t*>(attn_val_data);
        auto* out   = reinterpret_cast<llaisys::bf16_t*>(attn_val_data);
        
        //for (size_t sq = 0; sq < seqlen; sq++) {
        for (int64_t sq = 0; sq < seqlen; ++sq) {
            int64_t max_k = sq + diag;
            int64_t valid_len_i64 = max_k + 1;
            if (valid_len_i64 < 0) valid_len_i64 = 0;
            if (valid_len_i64 > total_len) valid_len_i64 = total_len;
            const int64_t valid_len = valid_len_i64;

            //for (size_t h = 0; h < nhead; h++) {
            for (int64_t h = 0; h < nhead; ++h) {
                // //size_t kv_head_idx = h / head_group;
                // // 使用取模运算，匹配 repeat_interleave
                // size_t kv_head_idx = h % nkvhead;
                const int64_t kv_head_idx = (head_group > 0) ? (h / head_group) : 0;
                
                const llaisys::bf16_t* q_head = q_ptr + (sq * nhead + h) * d;
                //llaisys::bf16_t* attn_head = attn_val_ptr + (sq * nhead + h) * dv;
                llaisys::bf16_t* out_head     = out   + (sq * nhead + h) * dv;
                
                // // 初始化注意力值为0
                // for (size_t dv_idx = 0; dv_idx < dv; dv_idx++) {
                //     //attn_head[dv_idx] = llaisys::bf16_t(0.0f);
                //     attn_head[dv_idx] = llaisys::utils::_f32_to_bf16(0.0f);
                // }

                float* acc = new float[(size_t)dv];
                for (int64_t i = 0; i < dv; ++i) acc[i] = 0.0f;

                if (valid_len == 0) {
                    for (int64_t i = 0; i < dv; ++i) out_head[i] = llaisys::utils::_f32_to_bf16(0.0f);
                    delete[] acc;
                    continue;
                }

                float* scores = new float[(size_t)valid_len];
                
                float max_score = -INFINITY;
            //     //size_t valid_len = (sq < total_len) ? sq + 1 : total_len;
            //     size_t valid_len = std::min(sq + 1, total_len);
            //     //float* attn_scores = new float[total_len];
            //     float* attn_scores = new float[valid_len];
                
            //     // 计算 Q·K^T
            //     //for (size_t tk = 0; tk <= sq; tk++) {
            //     for (size_t tk = 0; tk < valid_len; tk++) {
            //         const llaisys::bf16_t* k_head = k_ptr + (tk * nkvhead + kv_head_idx) * d;
                    
            //         float score = 0.0f;
            //         for (size_t idx = 0; idx < d; idx++) {
            //             float q_val = llaisys::utils::_bf16_to_f32(q_head[idx]);
            //             float k_val = llaisys::utils::_bf16_to_f32(k_head[idx]);
            //             score += q_val * k_val;
            //         }
            //         score *= scale;
                    
            //         // if (tk > sq) {
            //         //     score = -INFINITY;
            //         // }
                    
            //         attn_scores[tk] = score;
            //         if (score > max_score) {
            //             max_score = score;
            //         }
            //     }
                
            //     // 计算 softmax
            //     float exp_sum = 0.0f;
            //     //for (size_t tk = 0; tk <= sq; tk++) {
            //     for (size_t tk = 0; tk < valid_len; tk++) {
            //         float exp_val = std::exp(attn_scores[tk] - max_score);
            //         exp_sum += exp_val;
            //         attn_scores[tk] = exp_val;
            //     }
                
            //     // 加权求和
            //     //for (size_t tk = 0; tk <= sq; tk++) {
            //     for (size_t tk = 0; tk < valid_len; tk++) {
            //         float weight = attn_scores[tk] / exp_sum;
            //         const llaisys::bf16_t* v_head = v_ptr + (tk * nkvhead + kv_head_idx) * dv;
                    
            //         for (size_t dv_idx = 0; dv_idx < dv; dv_idx++) {
            //             float attn_val_f = llaisys::utils::_bf16_to_f32(attn_head[dv_idx]);
            //             float v_val = llaisys::utils::_bf16_to_f32(v_head[dv_idx]);
            //             attn_val_f += weight * v_val;
            //             attn_head[dv_idx] = llaisys::utils::_f32_to_bf16(attn_val_f);
            //         }
            //     }
                
            //     delete[] attn_scores;
            // }
            for (int64_t tk = 0; tk < valid_len; ++tk) {
                    const llaisys::bf16_t* k_head = k_ptr + (tk * nkvhead + kv_head_idx) * d;

                    float dot = 0.0f;
                    for (int64_t i = 0; i < d; ++i) {
                        float qv = llaisys::utils::_bf16_to_f32(q_head[i]);
                        float kv = llaisys::utils::_bf16_to_f32(k_head[i]);
                        dot += qv * kv;
                    }

                    float s = dot * scale;
                    scores[tk] = s;
                    if (s > max_score) max_score = s;
                }

                float exp_sum = 0.0f;
                for (int64_t tk = 0; tk < valid_len; ++tk) {
                    float e = std::exp(scores[tk] - max_score);
                    scores[tk] = e;
                    exp_sum += e;
                }

                const float inv_sum = 1.0f / exp_sum;
                for (int64_t tk = 0; tk < valid_len; ++tk) {
                    const float w = scores[tk] * inv_sum;
                    const llaisys::bf16_t* v_head = v_ptr + (tk * nkvhead + kv_head_idx) * dv;
                    for (int64_t i = 0; i < dv; ++i) {
                        float vv = llaisys::utils::_bf16_to_f32(v_head[i]);
                        acc[i] += w * vv;
                    }
                }

                for (int64_t i = 0; i < dv; ++i) out_head[i] = llaisys::utils::_f32_to_bf16(acc[i]);

                delete[] scores;
                delete[] acc;
            }
        }
        break;
    }
    
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(static_cast<llaisysDataType_t>(dtype));
    }

}
} // namespace llaisys::ops
