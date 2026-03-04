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
    // ... [此处省略所有原有检查代码，完全不变] ...

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