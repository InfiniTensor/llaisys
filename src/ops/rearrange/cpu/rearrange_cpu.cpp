#include "rearrange_cpu.hpp"
#include "../../../utils.hpp"

template <typename T>
void rearrange_(T *out, const T *in, const std::vector<size_t>& shape, 
                const std::vector<ptrdiff_t>& in_strides, 
                const std::vector<ptrdiff_t>& out_strides) {
    size_t rank = shape.size();
    size_t total_elements = 1;
    for (auto s : shape) total_elements *= s;

    std::vector<size_t> current_idx(rank, 0);

    for (size_t i = 0; i < total_elements; ++i) {
        // 使用 ptrdiff_t 计算偏移，以匹配传入的步长类型
        ptrdiff_t in_offset = 0;
        ptrdiff_t out_offset = 0;
        for (size_t d = 0; d < rank; ++d) {
            in_offset += static_cast<ptrdiff_t>(current_idx[d]) * in_strides[d];
            out_offset += static_cast<ptrdiff_t>(current_idx[d]) * out_strides[d];
        }

        out[out_offset] = in[in_offset];

        for (int d = static_cast<int>(rank) - 1; d >= 0; --d) {
            current_idx[d]++;
            if (current_idx[d] < shape[d]) break;
            current_idx[d] = 0;
        }
    }
}

namespace llaisys::ops::cpu {
void rearrange(std::byte *out, const std::byte *in, llaisysDataType_t type,
               const std::vector<size_t>& shape, 
               const std::vector<ptrdiff_t>& in_strides, 
               const std::vector<ptrdiff_t>& out_strides) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        rearrange_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), shape, in_strides, out_strides);
        break;
    case LLAISYS_DTYPE_BF16:
        rearrange_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in), shape, in_strides, out_strides);
        break;
    case LLAISYS_DTYPE_F16:
        rearrange_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in), shape, in_strides, out_strides);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
}