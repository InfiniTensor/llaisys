#include "embedding_cpu.hpp"
#include "../../../utils.hpp"
#include <cstddef>
#include <cstring>

namespace llaisys::ops::cpu {

template <typename T>

// 输入 weight: 形状为 [num_embeddings, embedding_dim] 的 2D 矩阵。
// 输入 index: 形状为 [seq_len] 的 1D 索引数组。
// 输出 out: 形状为 [seq_len, embedding_dim] 的 2D 矩阵
void embedding_kernel(const T *weight, const int64_t *index, T *out,
                      size_t seq_len, size_t embedding_dim) {
    for (size_t i = 0; i < seq_len; ++i) {
        int64_t idx = index[i]; // 获取当前位置需要的索引
        // 如果 p 是一个 T* 类型的指针，那么 p + n 实际上指向的内存地址是：数值地址(p) + n * sizeof(T)。
        // 从 src 指向的地址开始，连续复制 embedding_dim * sizeof(T) 个字节的数据，到以 dst 为起点的内存区域
        const T *src = weight + idx * embedding_dim;

        // 计算目的地址：out 矩阵的第 i 行
        T *dst = out + i * embedding_dim;
        
        // 执行内存拷贝：拷贝一整行数据
        // 从 src 指向的地址开始，连续复制 embedding_dim * sizeof(T) 个字节的数据，到以 dst 为起点的内存区域
        std::memcpy(dst, src, embedding_dim * sizeof(T));
    }
}

void embedding(const std::byte *weight, const std::byte *index, std::byte *out,
               llaisysDataType_t val_dtype, size_t seq_len, size_t embedding_dim) {
    // 右半部分：reinterpret_cast<const int64_t *>:
    // reinterpret_cast: 这是 C++ 中最强力的转换符。它的字面意思是“重新解释”。
    // 它告诉编译器：“我知道这块内存地址原本被标记为 std::byte（原始字节），但现在请把它当做 const int64_t 类型的数组来对待。”
    // <const int64_t *>: 这是转换的目标类型。
    //      尖括号 < > 里面填的就是你希望转换成的样子。这里要求转换成一个“指向 64 位常量整数的指针”。
    // (index): 这是转换的源对象。在你的代码中，index 原本的类型是 const std::byte *

    const int64_t *idx_ptr = reinterpret_cast<const int64_t *>(index);

    switch (val_dtype) {
        // switch (val_dtype): 这是该函数的核心。它检查传入的 val_dtype 枚举值。
        // 模板实例化: 对于每种支持的类型（F32, F16, BF16），它会实例化并调用 embedding_kernel<T>。
    case LLAISYS_DTYPE_F32:
        embedding_kernel<float>(reinterpret_cast<const float *>(weight), idx_ptr,
                                reinterpret_cast<float *>(out), seq_len, embedding_dim);
        break;
    case LLAISYS_DTYPE_F16:
        embedding_kernel<llaisys::fp16_t>(reinterpret_cast<const llaisys::fp16_t *>(weight), idx_ptr,
                                 reinterpret_cast<llaisys::fp16_t *>(out), seq_len, embedding_dim);
        break;
    case LLAISYS_DTYPE_BF16:
        embedding_kernel<llaisys::bf16_t>(reinterpret_cast<const llaisys::bf16_t *>(weight), idx_ptr,
                                 reinterpret_cast<llaisys::bf16_t *>(out), seq_len, embedding_dim);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(val_dtype);
    }
}

} // namespace llaisys::ops::cpu