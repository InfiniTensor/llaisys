#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include <cmath>
#include <limits>

#include <vector>

namespace {
template <typename T>
void self_attention_impl(T *attn_val_ptr, const T *q_ptr, const T *k_ptr, const T *v_ptr,
                         size_t seqlen, size_t nhead, size_t d,
                         size_t total_len, size_t nkvhead, size_t dv,
                         float scale) {
    size_t group_size = nhead / nkvhead; // GQA: 每个KV头对应group_size个Q头

    // 为每个序列位置计算attention
    for (size_t s = 0; s < seqlen; ++s) {
        // 为每个Q头计算attention
        for (size_t h = 0; h < nhead; ++h) {
            size_t kv_head = h / group_size; // GQA头映射，每group_size个Q头对应同一个KV头

            // Step 1: 计算所有位置的attention scores: Q[s,h] @ K[:,kv_head]^T
            std::vector<float> scores(total_len);                      // 用来存储当前Q头与所有其他词条其他相对应的K位置的得分
            float max_score = -std::numeric_limits<float>::infinity(); // 初始化用于softmax的最大值

            for (size_t t = 0; t < total_len; ++t) {
                float dot = 0.0f;
                for (size_t i = 0; i < d; ++i) {
                    float q_val = llaisys::utils::cast<float>(q_ptr[(s * nhead + h) * d + i]);
                    float k_val = llaisys::utils::cast<float>(k_ptr[(t * nkvhead + kv_head) * d + i]);
                    dot += q_val * k_val;
                }
                scores[t] = dot * scale;
                max_score = std::max(max_score, scores[t]);
            }

            // Step 2: 应用因果掩码
            size_t visible_len = s + (total_len - seqlen) + 1;
            for (size_t t = visible_len; t < total_len; ++t) {
                scores[t] = -std::numeric_limits<float>::infinity();
            }

            // Step 3: Softmax（数值稳定版本）
            float sum_exp = 0.0f;
            for (size_t t = 0; t < total_len; ++t) {
                if (std::isinf(scores[t]) && scores[t] < 0) {
                    scores[t] = 0.0f; // -inf会被exp为0
                } else {
                    scores[t] = std::exp(scores[t] - max_score);
                    sum_exp += scores[t];
                }
            }

            // 防止除零
            if (sum_exp == 0.0f) {
                sum_exp = 1.0f; // 避免NaN
            }

            // 归一化
            for (size_t t = 0; t < total_len; ++t) {
                scores[t] /= sum_exp;
            }

            // Step 4: 加权聚合V
            for (size_t j = 0; j < dv; ++j) {
                float output_val = 0.0f;
                for (size_t t = 0; t < total_len; ++t) {
                    float v_val = llaisys::utils::cast<float>(v_ptr[(t * nkvhead + kv_head) * dv + j]);
                    output_val += scores[t] * v_val;
                }
                attn_val_ptr[(s * nhead + h) * dv + j] = llaisys::utils::cast<T>(output_val);
            }
        }
    }
}
} // namespace

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    // 检查数据和设备类型
    CHECK_SAME_DEVICE(attn_val, q, k, v);

    auto dtype = q->dtype();
    ASSERT(attn_val->dtype() == dtype && k->dtype() == dtype && v->dtype() == dtype,
           "SelfAttention: all tensors must have the same dtype.");
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "SelfAttention: all tensors must be contiguous.");

    // 检查维度
    CHECK_ARGUMENT(attn_val->ndim() == 3, "SelfAttention: attn_val must be 3D.");
    CHECK_ARGUMENT(q->ndim() == 3, "SelfAttention: q must be 3D.");
    CHECK_ARGUMENT(k->ndim() == 3, "SelfAttention: k must be 3D.");
    CHECK_ARGUMENT(v->ndim() == 3, "SelfAttention: v must be 3D.");

    // 提取维度
    size_t seqlen = q->shape()[0];
    size_t nhead = q->shape()[1];
    size_t d = q->shape()[2];
    size_t total_len = k->shape()[0];
    size_t nkvhead = k->shape()[1];
    size_t dv = v->shape()[2];

    // 检查形状匹配
    // 序列长度：Q的序列长度应该 <= K/V的总长度（K/V可能包含缓存）
    CHECK_ARGUMENT(seqlen <= total_len,
                   "SelfAttention: q.shape[0] must be <= k.shape[0].");

    // Q/K/V的特征维度应该相同
    CHECK_ARGUMENT(k->shape()[2] == d,
                   "SelfAttention: k.shape[2] must equal q.shape[2].");
    CHECK_ARGUMENT(v->shape()[0] == total_len,
                   "SelfAttention: v.shape[0] must equal k.shape[0].");
    CHECK_ARGUMENT(v->shape()[1] == nkvhead,
                   "SelfAttention: v.shape[1] must equal k.shape[1].");

    // 输出张量形状验证
    CHECK_ARGUMENT(attn_val->shape()[0] == seqlen,
                   "SelfAttention: attn_val.shape[0] must equal q.shape[0].");
    CHECK_ARGUMENT(attn_val->shape()[1] == nhead,
                   "SelfAttention: attn_val.shape[1] must equal q.shape[1].");
    CHECK_ARGUMENT(attn_val->shape()[2] == dv,
                   "SelfAttention: attn_val.shape[2] must equal v.shape[2].");

    // GQA验证：Q头数必须能被KV头数整除
    CHECK_ARGUMENT(nhead % nkvhead == 0,
                   "SelfAttention: q.shape[1] (nhead) must be divisible by k.shape[1] (nkvhead).");

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        if (attn_val->deviceType() == LLAISYS_DEVICE_NVIDIA) {
            auto &ctx = llaisys::core::context();
            ctx.setDevice(attn_val->deviceType(), attn_val->deviceId());
            const auto *api = ctx.runtime().api();

            const size_t out_bytes = attn_val->numel() * attn_val->elementSize();
            const size_t q_bytes = q->numel() * q->elementSize();
            const size_t k_bytes = k->numel() * k->elementSize();
            const size_t v_bytes = v->numel() * v->elementSize();
            std::vector<std::byte> h_out(out_bytes), h_q(q_bytes), h_k(k_bytes), h_v(v_bytes);

            api->memcpy_sync(h_q.data(), q->data(), q_bytes, LLAISYS_MEMCPY_D2H);
            api->memcpy_sync(h_k.data(), k->data(), k_bytes, LLAISYS_MEMCPY_D2H);
            api->memcpy_sync(h_v.data(), v->data(), v_bytes, LLAISYS_MEMCPY_D2H);

            self_attention_impl(reinterpret_cast<float *>(h_out.data()),
                                reinterpret_cast<const float *>(h_q.data()),
                                reinterpret_cast<const float *>(h_k.data()),
                                reinterpret_cast<const float *>(h_v.data()),
                                seqlen, nhead, d, total_len, nkvhead, dv, scale);
            api->memcpy_sync(attn_val->data(), h_out.data(), out_bytes, LLAISYS_MEMCPY_H2D);
            return;
        }
        if (attn_val->deviceType() != LLAISYS_DEVICE_CPU) {
            EXCEPTION_UNSUPPORTED_DEVICE;
        }
        return self_attention_impl(reinterpret_cast<float *>(attn_val->data()),
                                   reinterpret_cast<const float *>(q->data()),
                                   reinterpret_cast<const float *>(k->data()),
                                   reinterpret_cast<const float *>(v->data()),
                                   seqlen, nhead, d, total_len, nkvhead, dv, scale);

    case LLAISYS_DTYPE_F16:
        if (attn_val->deviceType() == LLAISYS_DEVICE_NVIDIA) {
            auto &ctx = llaisys::core::context();
            ctx.setDevice(attn_val->deviceType(), attn_val->deviceId());
            const auto *api = ctx.runtime().api();

            const size_t out_bytes = attn_val->numel() * attn_val->elementSize();
            const size_t q_bytes = q->numel() * q->elementSize();
            const size_t k_bytes = k->numel() * k->elementSize();
            const size_t v_bytes = v->numel() * v->elementSize();
            std::vector<std::byte> h_out(out_bytes), h_q(q_bytes), h_k(k_bytes), h_v(v_bytes);

            api->memcpy_sync(h_q.data(), q->data(), q_bytes, LLAISYS_MEMCPY_D2H);
            api->memcpy_sync(h_k.data(), k->data(), k_bytes, LLAISYS_MEMCPY_D2H);
            api->memcpy_sync(h_v.data(), v->data(), v_bytes, LLAISYS_MEMCPY_D2H);

            self_attention_impl(reinterpret_cast<llaisys::fp16_t *>(h_out.data()),
                                reinterpret_cast<const llaisys::fp16_t *>(h_q.data()),
                                reinterpret_cast<const llaisys::fp16_t *>(h_k.data()),
                                reinterpret_cast<const llaisys::fp16_t *>(h_v.data()),
                                seqlen, nhead, d, total_len, nkvhead, dv, scale);
            api->memcpy_sync(attn_val->data(), h_out.data(), out_bytes, LLAISYS_MEMCPY_H2D);
            return;
        }
        if (attn_val->deviceType() != LLAISYS_DEVICE_CPU) {
            EXCEPTION_UNSUPPORTED_DEVICE;
        }
        return self_attention_impl(reinterpret_cast<llaisys::fp16_t *>(attn_val->data()),
                                   reinterpret_cast<const llaisys::fp16_t *>(q->data()),
                                   reinterpret_cast<const llaisys::fp16_t *>(k->data()),
                                   reinterpret_cast<const llaisys::fp16_t *>(v->data()),
                                   seqlen, nhead, d, total_len, nkvhead, dv, scale);

    case LLAISYS_DTYPE_BF16:
        if (attn_val->deviceType() == LLAISYS_DEVICE_NVIDIA) {
            auto &ctx = llaisys::core::context();
            ctx.setDevice(attn_val->deviceType(), attn_val->deviceId());
            const auto *api = ctx.runtime().api();

            const size_t out_bytes = attn_val->numel() * attn_val->elementSize();
            const size_t q_bytes = q->numel() * q->elementSize();
            const size_t k_bytes = k->numel() * k->elementSize();
            const size_t v_bytes = v->numel() * v->elementSize();
            std::vector<std::byte> h_out(out_bytes), h_q(q_bytes), h_k(k_bytes), h_v(v_bytes);

            api->memcpy_sync(h_q.data(), q->data(), q_bytes, LLAISYS_MEMCPY_D2H);
            api->memcpy_sync(h_k.data(), k->data(), k_bytes, LLAISYS_MEMCPY_D2H);
            api->memcpy_sync(h_v.data(), v->data(), v_bytes, LLAISYS_MEMCPY_D2H);

            self_attention_impl(reinterpret_cast<llaisys::bf16_t *>(h_out.data()),
                                reinterpret_cast<const llaisys::bf16_t *>(h_q.data()),
                                reinterpret_cast<const llaisys::bf16_t *>(h_k.data()),
                                reinterpret_cast<const llaisys::bf16_t *>(h_v.data()),
                                seqlen, nhead, d, total_len, nkvhead, dv, scale);
            api->memcpy_sync(attn_val->data(), h_out.data(), out_bytes, LLAISYS_MEMCPY_H2D);
            return;
        }
        if (attn_val->deviceType() != LLAISYS_DEVICE_CPU) {
            EXCEPTION_UNSUPPORTED_DEVICE;
        }
        return self_attention_impl(reinterpret_cast<llaisys::bf16_t *>(attn_val->data()),
                                   reinterpret_cast<const llaisys::bf16_t *>(q->data()),
                                   reinterpret_cast<const llaisys::bf16_t *>(k->data()),
                                   reinterpret_cast<const llaisys::bf16_t *>(v->data()),
                                   seqlen, nhead, d, total_len, nkvhead, dv, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops