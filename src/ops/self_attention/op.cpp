#include "op.hpp"
#include <cmath>
#include <cstring>
namespace llaisys::ops {

void self_attention(tensor_t out, tensor_t q, tensor_t k, tensor_t v, float scale) {
    auto get_float_at = [&](const std::byte* data, size_t elem_offset, llaisysDataType_t dtype) -> float {
        const std::byte* ptr = data + elem_offset * utils::dsize(dtype);
        switch (dtype) {
            case LLAISYS_DTYPE_F32: { float val; std::memcpy(&val, ptr, sizeof(float)); return val; }
            case LLAISYS_DTYPE_F16: { fp16_t val; std::memcpy(&val, ptr, sizeof(fp16_t)); return utils::cast<float>(val); }
            case LLAISYS_DTYPE_BF16: { bf16_t val; std::memcpy(&val, ptr, sizeof(bf16_t)); return utils::cast<float>(val); }
            default: EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
        }
    };

    auto set_float_at = [&](std::byte* data, size_t elem_offset, float val, llaisysDataType_t dtype) {
        std::byte* ptr = data + elem_offset * utils::dsize(dtype);
        switch (dtype) {
            case LLAISYS_DTYPE_F32: { std::memcpy(ptr, &val, sizeof(float)); break; }
            case LLAISYS_DTYPE_F16: { fp16_t h = utils::cast<fp16_t>(val); std::memcpy(ptr, &h, sizeof(fp16_t)); break; }
            case LLAISYS_DTYPE_BF16: { bf16_t b = utils::cast<bf16_t>(val); std::memcpy(ptr, &b, sizeof(bf16_t)); break; }
            default: EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
        }
    };

    size_t seqlen = q->shape()[0];
    size_t total_len = k->shape()[0];
    size_t nhead = q->shape()[1];
    size_t nkvhead = k->shape()[1];
    size_t d = q->shape()[2];
    size_t dv = v->shape()[2];
    size_t group_size = nhead / nkvhead;

    for (size_t i = 0; i < seqlen; ++i) {
        size_t query_global_pos = (total_len - seqlen) + i;
        for (size_t h = 0; h < nhead; ++h) {
            size_t kv_h = h / group_size;
            std::vector<float> logits(total_len);
            float max_score = -std::numeric_limits<float>::infinity();
            for (size_t t = 0; t < total_len; ++t) {
                if (t > query_global_pos) {
                    logits[t] = -std::numeric_limits<float>::infinity();
                    continue;
                }
                float score = 0.0f;
                for (size_t j = 0; j < d; ++j) {
                    score += get_float_at(q->data(), i * nhead * d + h * d + j, q->dtype()) *
                             get_float_at(k->data(), t * nkvhead * d + kv_h * d + j, k->dtype());
                }
                score *= scale;
                logits[t] = score;
                if (score > max_score) max_score = score;
            }

            float sum_exp = 0.0f;
            for (size_t t = 0; t < total_len; ++t) {
                if (logits[t] == -std::numeric_limits<float>::infinity()) {
                    logits[t] = 0.0f;
                } else {
                    float exp_val = std::exp(logits[t] - max_score);
                    logits[t] = exp_val;
                    sum_exp += exp_val;
                }
            }
            float inv_sum = 1.0f / (sum_exp + 1e-9f);

            std::vector<float> acc_out(dv, 0.0f);
            for (size_t t = 0; t < total_len; ++t) {
                if (logits[t] == 0.0f) continue;
                float prob = logits[t] * inv_sum;
                for (size_t j = 0; j < dv; ++j) {
                    acc_out[j] += prob * get_float_at(v->data(), t * nkvhead * dv + kv_h * dv + j, v->dtype());
                }
            }

            for (size_t j = 0; j < dv; ++j) {
                set_float_at(out->data(), i * nhead * dv + h * dv + j, acc_out[j], out->dtype());
            }
        }
    }
}

} // namespace llaisys::ops