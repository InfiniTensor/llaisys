#include "op.hpp"
#include "../../utils.hpp"
#include "./cpu/embedding_cpu.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    // 1. 基础校验
    CHECK_SAME_DEVICE(out, index, weight); // 设备一致性
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype()); // 类型一致性
    CHECK_ARGUMENT(index->dtype() == LLAISYS_DTYPE_I64, "Embedding index must be Int64."); // 索引类型约束：明确要求 index 必须是 Int64 类型

    // 2. 形状校验
    // weight: [num_embeddings, embedding_dim]
    // index: [seq_len]
    // out: [seq_len, embedding_dim]
    CHECK_ARGUMENT(weight->ndim() == 2, "Weight must be 2D.");
    CHECK_ARGUMENT(index->ndim() == 1, "Index must be 1D.");
    CHECK_ARGUMENT(out->ndim() == 2, "Output must be 2D.");
    
    size_t seq_len = index->numel();
    size_t embedding_dim = weight->shape()[1];
    
    CHECK_ARGUMENT(out->shape()[0] == seq_len, "Output shape[0] mismatch with index length.");
    CHECK_ARGUMENT(out->shape()[1] == embedding_dim, "Output shape[1] mismatch with embedding_dim.");

    // 3. 内存连续性校验
    // 底层的 CPU 内核（见 embedding_cpu.cpp）使用了简单的指针算术：weight + idx * embedding_dim。
    // 这种计算方式仅在内存连续排列时有效。如果张量经过了 slice 或 permute 导致步长（strides）不标准，
    // 直接计算地址会指向错误的数据，因此这里强制要求连续性。
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(),
           "Embedding: all tensors must be contiguous.");

    // 4. CPU 分发
    // 解包：它通过 data() 函数（定义在 tensor.cpp 中）将高层的 Tensor 对象解包为底层的 std::byte* 原始指针。
    // 类型擦除：注意这里传递的是原始指针和 dtype 枚举，具体的类型识别（如 float vs fp16）将由底层内核通过 switch-case 完成
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        cpu::embedding(weight->data(), index->data(), out->data(),
                       weight->dtype(), seq_len, embedding_dim);
        return;
    }

    // 5. GPU 分支待实现
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    TO_BE_IMPLEMENTED();
}
} // namespace llaisys::ops
