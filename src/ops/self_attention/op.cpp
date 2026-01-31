#include "op.hpp"
#include <cmath>
#include <cstring>

namespace llaisys::ops {

void self_attention(tensor_t out, tensor_t q, tensor_t k, tensor_t v, float scale) {
    auto get_float_at = [&](const std::byte* data, size_t elem_offset, llaisysDataType_t dtype) -> float {
        const std::byte* ptr = data + elem_offset * utils::dsize(dtype);
        switch (dtype) {
            case LLAISYS_DTYPE_F32: {
                float val; std::memcpy(&val, ptr, sizeof(float)); return val;
            }
            case LLAISYS_DTYPE_F16: {
                fp16_t val; std::memcpy(&val, ptr, sizeof(fp16_t)); return utils::cast<float>(val);
            }
            case LLAISYS_DTYPE_BF16: {
                bf16_t val; std::memcpy(&val, ptr, sizeof(bf16_t)); return utils::cast<float>(val);
            }
            default:
                EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
        }
    };
    auto set_float_at = [&](std::byte* data, size_t elem_offset, float val, llaisysDataType_t dtype) {
        std::byte* ptr = data + elem_offset * utils::dsize(dtype);
        switch (dtype) {
            case LLAISYS_DTYPE_F32: {
                std::memcpy(ptr, &val, sizeof(float)); break;
            }
            case LLAISYS_DTYPE_F16: {
                fp16_t h = utils::cast<fp16_t>(val);
                std::memcpy(ptr, &h, sizeof(fp16_t)); break;
            }
            case LLAISYS_DTYPE_BF16: {
                bf16_t b = utils::cast<bf16_t>(val);
                std::memcpy(ptr, &b, sizeof(bf16_t)); break;
            }
            default:
                EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
        }
    };
    size_t seqlen   = q->shape()[0];
    size_t nhead    = q->shape()[1];
    size_t d        = q->shape()[2];
    size_t total_len = k->shape()[0];
    size_t nkvhead   = k->shape()[1];
    size_t dv        = v->shape()[2];
    if (nhead % nkvhead != 0) {
        throw std::runtime_error("nhead must be divisible by nkvhead");
    }
    size_t group_size = nhead / nkvhead;
    float*** A = new float**[seqlen];
    for (size_t i = 0; i < seqlen; ++i) {
        A[i] = new float*[nhead];
        for (size_t h = 0; h < nhead; ++h) {
            A[i][h] = new float[total_len];
        }
    }
    for (size_t i = 0; i < seqlen; ++i) {
        for (size_t h = 0; h < nhead; ++h) {
            size_t kvh = h / group_size;
            for (size_t j = 0; j < total_len; ++j) {
                float sum = 0.0f;
                for (size_t idx = 0; idx < d; ++idx) {
                    size_t q_offset = i * nhead * d + h * d + idx;
                    size_t k_offset = j * nkvhead * d + kvh * d + idx;
                    float q_val = get_float_at(q->data(), q_offset, q->dtype());
                    float k_val = get_float_at(k->data(), k_offset, k->dtype());
                    sum += q_val * k_val;
                }
                A[i][h][j] = sum * scale;
            }
        }
    }
    size_t base_pos = total_len - seqlen; 
    for (size_t i = 0; i < seqlen; ++i) {
        for (size_t h = 0; h < nhead; ++h) {
            for (size_t j = 0; j < total_len; ++j) {
                if (j > base_pos + i) {
                    A[i][h][j] = -INFINITY;
                }
            }
            float max_val = -INFINITY;
            for (size_t j = 0; j <= base_pos + i; ++j)
                if (A[i][h][j] > max_val) max_val = A[i][h][j];
            float exp_sum = 0.0f;
            for (size_t j = 0; j <= base_pos + i; ++j) 
                exp_sum += expf(A[i][h][j] - max_val);
            for (size_t j = 0; j <= base_pos + i; ++j)
                A[i][h][j] = expf(A[i][h][j] - max_val) / exp_sum;
        }
    }
    for (size_t i = 0; i < seqlen; ++i) {
        for (size_t h = 0; h < nhead; ++h) {
            size_t kvh = h / group_size;
            for (size_t l = 0; l < dv; ++l) {
                float y_val = 0.0f;
                for (size_t j = 0; j < total_len; ++j) {
                    size_t v_offset = j * nkvhead * dv + kvh * dv + l;
                    float v_val = get_float_at(v->data(), v_offset, v->dtype());
                    y_val += A[i][h][j] * v_val;
                }
                size_t out_offset = i * nhead * dv + h * dv + l;
                set_float_at(out->data(), out_offset, y_val, out->dtype());
            }
        }
    }
    for (size_t i = 0; i < seqlen; ++i) {
        for (size_t h = 0; h < nhead; ++h) 
            delete[] A[i][h];
        delete[] A[i];
    }
    delete[] A;
}

} // namespace llaisys::ops