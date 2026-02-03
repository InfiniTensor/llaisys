#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"

namespace llaisys::ops {

// 执行嵌入查找操作：根据索引张量从嵌入表中提取对应向量
void embedding(tensor_t output, tensor_t indices, tensor_t embedding_table) {
    // 确保所有张量位于同一设备
    CHECK_SAME_DEVICE(output, indices, embedding_table);
    // 输出与嵌入表必须具有相同的数据类型
    CHECK_SAME_DTYPE(output->dtype(), embedding_table->dtype());
    // 索引张量必须为 int64 类型
    ASSERT(indices->dtype() == LLAISYS_DTYPE_I64, "Embedding: index tensor must be of type int64.");
    // 索引必须是一维的
    ASSERT(indices->ndim() == 1, "Embedding: index tensor must be 1-dimensional.");
    // 嵌入表必须是二维的（词汇表大小 × 嵌入维度）
    ASSERT(embedding_table->ndim() == 2, "Embedding: embedding table must be 2-dimensional.");
    // 输出张量必须是二维的
    ASSERT(output->ndim() == 2, "Embedding: output tensor must be 2-dimensional.");

    const auto &table_shape = embedding_table->shape();
    size_t vocab_size = table_shape[0];
    size_t embedding_dim = table_shape[1];
    size_t num_indices = indices->numel();

    // 验证输出形状是否匹配：(num_indices, embedding_dim)
    ASSERT(output->shape()[0] == num_indices && output->shape()[1] == embedding_dim,
           "Embedding: output tensor shape does not match expected (num_indices, embedding_dim).");

    // 所有张量必须内存连续
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
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace llaisys::ops