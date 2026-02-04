#include "op.hpp"
#include "../../utils.hpp"
#include <cstring>

namespace llaisys::ops {

// 辅助函数：递归搬运数据
// dst: 目标（连续内存）的当前指针
// src: 源（非连续内存）的当前指针
// shape: 剩余维度的形状
// strides: 源 Tensor 在这些维度上的步幅
// dim: 当前正在处理第几维
// element_size: 每个元素的字节数 (例如 float 是 4)
void rearrange_recursive(std::byte* dst, const std::byte* src, 
                         const std::vector<size_t>& shape, 
                         const std::vector<ptrdiff_t>& strides, 
                         int dim, size_t element_size) {
    
    // 递归终止条件：到达了最后一维
    if (dim == shape.size() - 1) {
        // 在最内层循环，我们把这一行的数据搬过去
        // 注意：源数据在这一行可能也不是连续的（比如切片过），所以还得一个一个搬
        ptrdiff_t stride = strides[dim];
        size_t count = shape[dim];

        if (stride == 1) {
            // 如果最后一维步幅是 1，说明这一小段是连续的，直接 memcpy 加速
            std::memcpy(dst, src, count * element_size);
        } else {
            // 如果步幅不是 1，只能一个一个搬
            for (size_t i = 0; i < count; ++i) {
                std::memcpy(dst + i * element_size, src + i * stride * element_size, element_size);
            }
        }
    } else {
        // 递归步骤：还没到最底层，继续往下钻
        size_t count = shape[dim];      // 这一维有多少个元素
        ptrdiff_t stride = strides[dim]; // 这一维跳一步跨多少
        
        // 计算下一层子 Tensor 的总大小（字节），以便移动 dst 指针
        // 因为 dst 是连续的，所以下一层的大小就是所有子维度乘积 * 元素大小
        size_t inner_size = element_size;
        for (size_t k = dim + 1; k < shape.size(); ++k) {
            inner_size *= shape[k];
        }

        for (size_t i = 0; i < count; ++i) {
            rearrange_recursive(
                dst + i * inner_size,           // dst 紧挨着往后排
                src + i * stride * element_size, // src 按照步幅跳跃
                shape, strides, dim + 1, element_size
            );
        }
    }
}

void rearrange(tensor_t out, tensor_t in) {
    // 1. 获取基本信息
    auto& shape = in->shape();
    auto& strides = in->strides();
    size_t element_size = in->elementSize();

    // 2. 获取原始指针（生肉 std::byte*）
    // 这样我们就不需要关心是 float 还是 int，统一按字节搬运
    std::byte* out_ptr = out->data();
    const std::byte* in_ptr = in->data();

    // 3. 开始递归搬运
    if (shape.size() == 0) {
        // 特殊情况：标量（0维 Tensor）
        std::memcpy(out_ptr, in_ptr, element_size);
    } else {
        rearrange_recursive(out_ptr, in_ptr, shape, strides, 0, element_size);
    }
}

} // namespace llaisys::ops