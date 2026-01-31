#include "op.hpp"
#include <cstring>

namespace llaisys::ops {

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    auto get_float_at = [&](const std::byte* data,size_t i,size_t j,size_t stride_i,size_t stride_j,llaisysDataType_t dtype) -> float {
        const std::byte* ptr = data + (i * stride_i + j * stride_j) * utils::dsize(dtype);
        switch (dtype) {
            case LLAISYS_DTYPE_F32: {
                float v;
                std::memcpy(&v, ptr, sizeof(v));
                return v;
            }
            case LLAISYS_DTYPE_F16: {
                fp16_t v;
                std::memcpy(&v, ptr, sizeof(v));
                return utils::cast<float>(v);
            }
            case LLAISYS_DTYPE_BF16: {
                bf16_t v;
                std::memcpy(&v, ptr, sizeof(v));
                return utils::cast<float>(v);
            }
            default:
                EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
        }
    };
    auto set_float_at = [&](std::byte* data,size_t i,size_t j,size_t stride_i,size_t stride_j,llaisysDataType_t dtype,float val) {
        std::byte* ptr = data + (i * stride_i + j * stride_j) * utils::dsize(dtype);
        switch (dtype) {
            case LLAISYS_DTYPE_F32: {
                std::memcpy(ptr, &val, sizeof(val));
                break;
            }
            case LLAISYS_DTYPE_F16: {
                fp16_t v = utils::cast<fp16_t>(val);
                std::memcpy(ptr, &v, sizeof(v));
                break;
            }
            case LLAISYS_DTYPE_BF16: {
                bf16_t v = utils::cast<bf16_t>(val);
                std::memcpy(ptr, &v, sizeof(v));
                break;
            }
            default:
                EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
        }
    };
    for (size_t i = 0; i < in->shape()[0]; ++i) {
        for (size_t j = 0; j < weight->shape()[0]; ++j) {
            float sum = 0.f;
            for (size_t k = 0; k < in->shape()[1]; ++k) {
                float x = get_float_at(in->data(),i, k,in->strides()[0], in->strides()[1],in->dtype());
                float w = get_float_at(weight->data(),j, k,weight->strides()[0], weight->strides()[1],weight->dtype());
                sum += x * w;
            }
            if (bias) {
                float b = get_float_at(bias->data(),0, j,bias->strides()[0], bias->strides()[1],bias->dtype());
                sum += b;
            }
            set_float_at(out->data(),i, j,out->strides()[0], out->strides()[1],out->dtype(),sum);
        }
    }
}

} // namespace llaisys::ops