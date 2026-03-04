#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/embedding_nvidia.hpp"
#endif

namespace llaisys::ops {

// 执行嵌入查找操作：根据索引张量从嵌入表中提取对应向量
void embedding(tensor_t output, tensor_t indices, tensor_t embedding_table) {
    // 确保输出、索引和嵌入表位于同一设备 (Indices 和 Table 通常在同一设备，Output 也需一致)
    CHECK_SAME_DEVICE(output, indices);
    CHECK_SAME_DEVICE(output, embedding_table);
    
    // 数据类型检查：Output 和 Table 必须一致，Indices 必须是整型 (通常为 int32 或 int64)
    CHECK_SAME_DTYPE(output->dtype(), embedding_table->dtype());
    ASSERT(indices->dtype() == LLAISYS_DTYPE_I32 || indices->dtype() == LLAISYS_DTYPE_I64,
           "Embedding: indices tensor must be int32 or int64.");

    // 张量维度约束
    ASSERT(indices->ndim() == 1, "Embedding: indices tensor must be 1-dimensional.");
    ASSERT(embedding_table->ndim() == 2, "Embedding: embedding_table tensor must be 2-dimensional [vocab_size, dim].");
    ASSERT(output->ndim() == 2, "Embedding: output tensor must be 2-dimensional [num_indices, dim].");

    size_t num_indices = indices->shape()[0];
    size_t vocab_size = embedding_table->shape()[0];
    size_t embedding_dim = embedding_table->shape()[1];

    // 验证输出形状
    ASSERT(output->shape()[0] == num_indices && output->shape()[1] == embedding_dim,
           "Embedding: output shape does not match [num_indices, embedding_dim].");
    
    // 验证索引范围 (可选，通常在 debug 模式开启，生产环境为了性能可能省略或由 kernel 内部处理)
    // ASSERT(...) 

    // 所有张量必须内存连续 (对于 Embedding 操作，输出和输入表通常需要连续，indices 也必须连续)
    ASSERT(output->isContiguous() && indices->isContiguous() && embedding_table->isContiguous(),
           "Embedding: all tensors must be contiguous in memory.");

    if (output->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(
            output->data(),
            indices->data(),
            embedding_table->data(),
            output->dtype(),
            num_indices,
            embedding_dim,
            vocab_size
        );
    }

    llaisys::core::context().setDevice(output->deviceType(), output->deviceId());

    switch (output->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(
            output->data(),
            indices->data(),
            embedding_table->data(),
            output->dtype(),
            num_indices,
            embedding_dim,
            vocab_size
        );
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        nvidia::embedding(
            output->data(),
            indices->data(),
            embedding_table->data(),
            output->dtype(),
            num_indices,
            embedding_dim,
            vocab_size
        );
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace llaisys::ops