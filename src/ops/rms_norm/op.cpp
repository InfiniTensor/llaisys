#include "op.hpp"
#include <cstring>
#include <cmath>

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    auto get_float_at=[&](const std::byte* data,size_t i,size_t j,size_t stride_i,size_t stride_j)->float{
        const std::byte* ptr=data+(i*stride_i+j*stride_j)*utils::dsize(in->dtype());
        switch(in->dtype()){
            case LLAISYS_DTYPE_F32:{
                float val;
                std::memcpy(&val,ptr,sizeof(float));
                return val;
            }
            case LLAISYS_DTYPE_F16:{
                fp16_t val;
                std::memcpy(&val, ptr, sizeof(fp16_t));
                return utils::cast<float>(val);
            }
            case LLAISYS_DTYPE_BF16:{
                bf16_t val;
                std::memcpy(&val, ptr, sizeof(bf16_t));
                return utils::cast<float>(val);
            }
            default:EXCEPTION_UNSUPPORTED_DATATYPE(in->dtype());
        }
    };

    auto set_float_at = [&](std::byte* data, size_t i, size_t j, size_t stride_i, size_t stride_j, float val) {
        std::byte* ptr = data + (i * stride_i + j * stride_j) * utils::dsize(out->dtype());
        switch (out->dtype()) {
            case LLAISYS_DTYPE_F32: {
                std::memcpy(ptr, &val, sizeof(float)); break;
            }
            case LLAISYS_DTYPE_F16: {
                fp16_t h = utils::cast<fp16_t>(val); std::memcpy(ptr, &h, sizeof(fp16_t)); break;
            }
            case LLAISYS_DTYPE_BF16: {
                bf16_t b = utils::cast<bf16_t>(val); std::memcpy(ptr, &b, sizeof(bf16_t)); break;
            }
            default:
                EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
        }
    };

    for (size_t i = 0; i < in->shape()[0]; ++i) {
        float sum_sq = 0.0f;
        for (size_t j = 0; j < in->shape()[1]; ++j) {
            float x_val = get_float_at(in->data(), i, j, in->strides()[0], in->strides()[1]);
            sum_sq += x_val * x_val;
        }

        float norm = sqrt(sum_sq / in->shape()[1] + eps);

        for (size_t j = 0; j < in->shape()[1]; ++j) {
            float w_val = get_float_at(weight->data(), 0, j, 1, 1); 
            float x_val = get_float_at(in->data(), i, j, in->strides()[0], in->strides()[1]);
            float y_val = w_val * x_val / norm;
            set_float_at(out->data(), i, j, out->strides()[0], out->strides()[1], y_val);
        }
    }
}
} // namespace llaisys::ops