#include "rearrange_cpu.hpp"
#include "../../../utils.hpp"
#include <cstring>

namespace llaisys::ops::cpu {

template <typename T>
void rearrange_recursive(T *out, const T *in, 
                         const std::vector<size_t> &shape,
                         const std::vector<ptrdiff_t> &out_strides,
                         const std::vector<ptrdiff_t> &in_strides,
                         size_t dim, ptrdiff_t out_offset, ptrdiff_t in_offset) {
    size_t len = shape[dim];
    ptrdiff_t os = out_strides[dim];
    ptrdiff_t is = in_strides[dim];

    if (dim == shape.size() - 1) {
        // Base case: 最内层维度
        if (os == 1 && is == 1) {
            // 优化: 如果两边都是连续的，直接 memcpy
            std::memcpy(out + out_offset, in + in_offset, len * sizeof(T));
        } else {
            for (size_t i = 0; i < len; ++i) {
                // 使用 utils 中的 cast 辅助函数（虽然这里类型相同，但保持一致性）
                out[out_offset + (ptrdiff_t)i * os] = llaisys::utils::cast<T>(in[in_offset + (ptrdiff_t)i * is]);
            }
        }
    } else {
        // Recursive step: 递归处理下一维
        for (size_t i = 0; i < len; ++i) {
            rearrange_recursive(out, in, shape, out_strides, in_strides, dim + 1,
                                out_offset + (ptrdiff_t)i * os, in_offset + (ptrdiff_t)i * is);
        }
    }
}

template <typename T>
void rearrange_kernel(T *out, const T *in, 
                      const std::vector<size_t> &shape,
                      const std::vector<ptrdiff_t> &out_strides,
                      const std::vector<ptrdiff_t> &in_strides) {
    if (shape.empty()) {
        // 处理标量情况
        out[0] = llaisys::utils::cast<T>(in[0]);
        return;
    }

    // 对最外层维度进行并行化
    size_t dim0 = shape[0];
    ptrdiff_t os0 = out_strides[0];
    ptrdiff_t is0 = in_strides[0];

    #pragma omp parallel for
    for (size_t i = 0; i < dim0; ++i) {
        if (shape.size() > 1) {
            rearrange_recursive(out, in, shape, out_strides, in_strides, 1, 
                                (ptrdiff_t)i * os0, (ptrdiff_t)i * is0);
        } else {
            // 1D 张量的情况
            out[(ptrdiff_t)i * os0] = llaisys::utils::cast<T>(in[(ptrdiff_t)i * is0]);
        }
    }
}

void rearrange(std::byte *out, const std::byte *in, 
               const std::vector<size_t> &shape,
               const std::vector<ptrdiff_t> &out_strides,
               const std::vector<ptrdiff_t> &in_strides,
               llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        rearrange_kernel((float *)out, (const float *)in, shape, out_strides, in_strides);
        break;
    case LLAISYS_DTYPE_F16:
        rearrange_kernel((llaisys::fp16_t *)out, (const llaisys::fp16_t *)in, shape, out_strides, in_strides);
        break;
    case LLAISYS_DTYPE_BF16:
        rearrange_kernel((llaisys::bf16_t *)out, (const llaisys::bf16_t *)in, shape, out_strides, in_strides);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu