#include "rearrange_cpu.hpp"

#include <cstring>

namespace {
// 递归实现张量数据重排：根据输入/输出步长将数据从源布局复制到目标布局
void rearrange_recursive(std::byte *output,
                         const std::byte *input,
                         const std::vector<size_t> &tensor_shape,
                         const std::vector<ptrdiff_t> &output_strides,
                         const std::vector<ptrdiff_t> &input_strides,
                         size_t element_byte_size,
                         size_t current_dim,
                         ptrdiff_t output_offset,
                         ptrdiff_t input_offset) {
    // 递归终止条件：已遍历所有维度，执行单个元素的复制
    if (current_dim == tensor_shape.size()) {
        std::memcpy(
            output + output_offset * element_byte_size,
            input + input_offset * element_byte_size,
            element_byte_size
        );
        return;
    }

    const size_t dim_size = tensor_shape[current_dim];
    const ptrdiff_t out_stride = output_strides[current_dim];
    const ptrdiff_t in_stride = input_strides[current_dim];

    // 遍历当前维度的所有索引位置
    for (size_t idx = 0; idx < dim_size; ++idx) {
        rearrange_recursive(
            output,
            input,
            tensor_shape,
            output_strides,
            input_strides,
            element_byte_size,
            current_dim + 1,
            output_offset + static_cast<ptrdiff_t>(idx) * out_stride,
            input_offset + static_cast<ptrdiff_t>(idx) * in_stride
        );
    }
}
} // namespace

namespace llaisys::ops::cpu {

// 在 CPU 上执行通用张量重排操作（支持任意 strides）
void rearrange(std::byte *output,
               const std::byte *input,
               const std::vector<size_t> &shape,
               const std::vector<ptrdiff_t> &output_strides,
               const std::vector<ptrdiff_t> &input_strides,
               size_t elem_size) {
    rearrange_recursive(
        output, input, shape, output_strides, input_strides, elem_size, 0, 0, 0
    );
}

} // namespace llaisys::ops::cpu