#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cstdint>
#include <cstring>

namespace llaisys::ops::cpu {

// 在 CPU 上执行嵌入查找操作：根据索引从权重表中提取对应向量
void embedding(std::byte *output, const std::byte *indices, const std::byte *embedding_table,
               llaisysDataType_t dtype, size_t num_indices, size_t embedding_dim, size_t table_rows) {
    // 确定每个嵌入元素的字节大小
    size_t element_byte_size = 0;
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
    case LLAISYS_DTYPE_F16:
    case LLAISYS_DTYPE_BF16:
        element_byte_size = llaisys::utils::dsize(dtype);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }

    const int64_t *index_data = reinterpret_cast<const int64_t *>(indices);
    const size_t bytes_per_row = embedding_dim * element_byte_size;

    // 遍历每个索引，从嵌入表中复制对应行到输出
    for (size_t pos = 0; pos < num_indices; ++pos) {
        int64_t idx = index_data[pos];
        // 检查索引是否在有效范围内 [0, table_rows)
        ASSERT(idx >= 0 && static_cast<size_t>(idx) < table_rows, "Embedding: index out of range.");

        const std::byte *source_row = embedding_table + static_cast<size_t>(idx) * bytes_per_row;
        std::byte *target_row = output + pos * bytes_per_row;
        std::memcpy(target_row, source_row, bytes_per_row);
    }
}

} // namespace llaisys::ops::cpu