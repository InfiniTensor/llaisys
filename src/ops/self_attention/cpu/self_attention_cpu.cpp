#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <vector>

namespace {
constexpr float NEG_INF = -1e9f;
}

template <typename T>
static void self_attention_(std::byte *attn_val,
                            const std::byte *q,
                            const std::byte *k,
                            const std::byte *v,
                            size_t seqlen,
                            size_t nhead,
                            size_t nkvhead,
                            size_t d,
                            size_t dv,
                            size_t total_len,
                            float scale) {
    const T *qT = reinterpret_cast<const T *>(q);
    const T *kT = reinterpret_cast<const T *>(k);
    const T *vT = reinterpret_cast<const T *>(v);
    T *outT = reinterpret_cast<T *>(attn_val);

    std::vector<float> scores(seqlen * total_len);

    // 遍历层级：head(头)--->seqlen(序列长度)
    for (size_t h = 0; h < nhead; ++h) {          
        const size_t kv_head = h * nkvhead / nhead;

        // 1. Scores: (seqlen, total_len), A[i,j] = scale * q[i,h,:] @ k[j,kv_head,:]
        for (size_t i = 0; i < seqlen; ++i) {                 // 遍历每个query位置
            for (size_t j = 0; j < total_len; ++j) {            // 遍历每个key位置
                float acc = 0.f;
                for (size_t kd = 0; kd < d; ++kd) {
                    float qv = llaisys::utils::cast<float>(qT[(i * nhead + h) * d + kd]);
                    float kv = llaisys::utils::cast<float>(kT[(j * nkvhead + kv_head) * d + kd]);
                    acc += qv * kv;
                }
                scores[i * total_len + j] = scale * acc;
            }
        }

        // 2. Causal: mask (i,j) when j > i + (total_len - seqlen)
        // 这是为了确保在推理时，模型只能看到当前位置之前的上下文，而不能看到未来的信息
        // total_len：kvcache的总长度 seqlen：当前序列的长度
        // diag = total_len - seqlen : 历史token的数量(也就是当前序列的起始位置)
        // 置为-INF而不是0，因为exp(0) = 1，会导致softmax结果不正确
        const ptrdiff_t diag = static_cast<ptrdiff_t>(total_len) - static_cast<ptrdiff_t>(seqlen);
        for (size_t i = 0; i < seqlen; ++i) {
            for (size_t j = 0; j < total_len; ++j) {
                // i：当前query在序列中的相对位置，j：当前key在KV Cache中的绝对位置
                if (static_cast<ptrdiff_t>(j) > static_cast<ptrdiff_t>(i) + diag)
                    scores[i * total_len + j] = NEG_INF;    // mask掉未来的位置
            }
        }

        // 3. 对每个query位置，计算softmax：softmax(scores[i,:])->attn[i,:]
        for (size_t i = 0; i < seqlen; ++i) {
            float *row = &scores[i * total_len];
            float row_max = row[0];
            for (size_t j = 1; j < total_len; ++j) {
                if (row[j] > row_max)
                    row_max = row[j];
            }
            float sum = 0.f;
            for (size_t j = 0; j < total_len; ++j) {
                row[j] = std::exp(row[j] - row_max);
                sum += row[j];
            }
            for (size_t j = 0; j < total_len; ++j)
                row[j] /= sum;
        }

        // 4. 用注意力分数对V进行加权求和：attn_val[i,h,:](1 * dv) = attn[i,:] (1 * total_len) @ v[:,kv_head,:] (total_len * dv)
        // scores[seqlen, total_len], v[total_len, nkvhead, dv], out[seqlen, nhead, dv]
        for (size_t i = 0; i < seqlen; ++i) {
            for (size_t m = 0; m < dv; ++m) {
                float acc = 0.f;
                for (size_t j = 0; j < total_len; ++j) {
                    acc += scores[i * total_len + j] * llaisys::utils::cast<float>(vT[(j * nkvhead + kv_head) * dv + m]);
                }
                outT[(i * nhead + h) * dv + m] = llaisys::utils::cast<T>(acc);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val,
                    const std::byte *q,
                    const std::byte *k,
                    const std::byte *v,
                    llaisysDataType_t dtype,
                    size_t seqlen,
                    size_t nhead,
                    size_t nkvhead,
                    size_t d,
                    size_t dv,
                    size_t total_len,
                    float scale) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return self_attention_<float>(attn_val, q, k, v, seqlen, nhead, nkvhead, d, dv, total_len, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_<llaisys::fp16_t>(attn_val, q, k, v, seqlen, nhead, nkvhead, d, dv, total_len, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_<llaisys::bf16_t>(attn_val, q, k, v, seqlen, nhead, nkvhead, d, dv, total_len, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
