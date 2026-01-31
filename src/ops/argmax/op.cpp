#include "op.hpp"
#include <cstring>
#include <limits>

namespace llaisys::ops {

void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    size_t nel = vals->numel();
    if (nel == 0)
        throw std::runtime_error("Cannot argmax on empty tensor");
    const std::byte* data = vals->data();
    auto strides = vals->strides();
    llaisysDataType_t dtype = vals->dtype();
    size_t elem_size = utils::dsize(dtype);
    std::byte* out_val_data = max_val->data();
    int64_t* out_idx = reinterpret_cast<int64_t*>(max_idx->data());
    float best = std::numeric_limits<float>::lowest();
    size_t best_i = 0;
    auto get_float_at = [&](size_t i) -> float {
        const std::byte* ptr = data + i * strides[0] * elem_size;
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
    for (size_t i = 0; i < nel; ++i) {
        float v = get_float_at(i);
        if (v > best) {
            best = v;
            best_i = i;
        }
    }
    out_idx[0] = static_cast<int64_t>(best_i);
    const std::byte* src = data + best_i * strides[0] * elem_size;
    std::memcpy(out_val_data, src, elem_size);
}

} // namespace llaisys::ops