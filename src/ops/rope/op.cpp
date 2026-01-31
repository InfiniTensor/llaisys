#include "op.hpp"
#include <cmath>
#include <cstring>
namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    auto get_float_at = [&](const std::byte* data, size_t elem_offset) -> float {
        const std::byte* ptr = data + elem_offset * utils::dsize(in->dtype());
        switch (in->dtype()) {
            case LLAISYS_DTYPE_F32: {
                float val; std::memcpy(&val, ptr, sizeof(float)); return val;
            }
            case LLAISYS_DTYPE_F16: {
                fp16_t val; std::memcpy(&val, ptr, sizeof(fp16_t)); return utils::cast<float>(val);
            }
            case LLAISYS_DTYPE_BF16: {
                bf16_t val; std::memcpy(&val, ptr, sizeof(bf16_t)); return utils::cast<float>(val);
            }
            default: EXCEPTION_UNSUPPORTED_DATATYPE(in->dtype());
        }
    };
    auto set_float_at = [&](std::byte* data, size_t elem_offset, float val) {
        std::byte* ptr = data + elem_offset * utils::dsize(out->dtype());
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
            default: EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
        }
    };
    auto get_int_at = [&](const std::byte* data, size_t elem_offset) -> int64_t {
        const std::byte* ptr = data + elem_offset * utils::dsize(pos_ids->dtype());
        switch (pos_ids->dtype()) {
            case LLAISYS_DTYPE_I64: {
                int64_t val; std::memcpy(&val, ptr, sizeof(int64_t)); return val;
            }
            case LLAISYS_DTYPE_I32: {
                int32_t val; std::memcpy(&val, ptr, sizeof(int32_t)); return val;
            }
            case LLAISYS_DTYPE_I16: {
                int16_t val; std::memcpy(&val, ptr, sizeof(int16_t)); return val;
            }
            case LLAISYS_DTYPE_I8: {
                int8_t val; std::memcpy(&val, ptr, sizeof(int8_t)); return val;
            }
            default: EXCEPTION_UNSUPPORTED_DATATYPE(pos_ids->dtype());
        }
    };
    for (size_t i = 0; i < in->shape()[0]; ++i) {
        for (size_t h = 0; h < in->shape()[1]; ++h) {
            for (size_t j = 0; j < in->shape()[2] / 2; ++j) {
                double angle = static_cast<double>(get_int_at(pos_ids->data(), i)) / std::pow(static_cast<double>(theta), (2.0 * static_cast<double>(j)) / static_cast<double>(in->shape()[2]));
                double cos_val = std::cos(angle);
                double sin_val = std::sin(angle);
                size_t base = i * in->shape()[1] * in->shape()[2] + h * in->shape()[2];
                size_t idx_a = base + j;
                size_t idx_b = base + j + in->shape()[2] / 2;
                float a = get_float_at(in->data(), idx_a);
                float b = get_float_at(in->data(), idx_b);
                set_float_at(out->data(), idx_a, static_cast<float>(static_cast<double>(a) * cos_val - static_cast<double>(b) * sin_val));
                set_float_at(out->data(), idx_b, static_cast<float>(static_cast<double>(b) * cos_val + static_cast<double>(a) * sin_val));
            }
        }
    }
}

} // namespace llaisys::ops