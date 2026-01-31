#include "op.hpp"
#include <cstring>
namespace llaisys::ops {

void add(tensor_t c, tensor_t a, tensor_t b) {
    auto get_value_at = [&](const std::byte* data, size_t elem_offset, llaisysDataType_t dtype) -> float {
        const std::byte* ptr = data + elem_offset * utils::dsize(dtype);
        switch (dtype) {
            case LLAISYS_DTYPE_F32: { float val; std::memcpy(&val, ptr, sizeof(float)); return val; }
            case LLAISYS_DTYPE_F16: { fp16_t val; std::memcpy(&val, ptr, sizeof(fp16_t)); return utils::cast<float>(val); }
            case LLAISYS_DTYPE_BF16: { bf16_t val; std::memcpy(&val, ptr, sizeof(bf16_t)); return utils::cast<float>(val); }
            default: EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
        }
    };

    auto set_value_at = [&](std::byte* data, size_t elem_offset, float val, llaisysDataType_t dtype) {
        std::byte* ptr = data + elem_offset * utils::dsize(dtype);
        switch (dtype) {
            case LLAISYS_DTYPE_F32: { std::memcpy(ptr, &val, sizeof(float)); break; }
            case LLAISYS_DTYPE_F16: { fp16_t h = utils::cast<fp16_t>(val); std::memcpy(ptr, &h, sizeof(fp16_t)); break; }
            case LLAISYS_DTYPE_BF16: { bf16_t b = utils::cast<bf16_t>(val); std::memcpy(ptr, &b, sizeof(bf16_t)); break; }
            default: EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
        }
    };

    size_t numel = a->shape()[0]; // Assuming all tensors have the same number of elements

    for (size_t i = 0; i < numel; ++i) {
        float a_val = get_value_at(a->data(), i, a->dtype());
        float b_val = get_value_at(b->data(), i, b->dtype());
        float result = a_val + b_val;
        set_value_at(c->data(), i, result, c->dtype());
    }
}

} // namespace llaisys::ops