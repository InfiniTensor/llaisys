#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/embedding_nvidia.hpp"
#endif

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype()); 
    
    // 🚨 修正：严格对齐测试脚本和 CPU 版本的 I64 类型
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "Embedding: index tensor must be I64.");
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(), "Embedding: all tensors must be contiguous.");
    
    size_t num_indices = index->numel();
    size_t vocab_size = weight->shape().front();
    size_t embedding_dim = weight->shape().back();

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), weight->data(), out->dtype(), num_indices, vocab_size, embedding_dim);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(), out->dtype(), num_indices, vocab_size, embedding_dim);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::embedding(out->data(), index->data(), weight->data(), out->dtype(), num_indices, vocab_size, embedding_dim);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops