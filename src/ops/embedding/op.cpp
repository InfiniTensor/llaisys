#include "op.hpp"

#include "cpu/embedding_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "../nvidia/ops_nvidia.cuh"
#endif

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    if (index->dtype() != LLAISYS_DTYPE_I64) {
        throw std::runtime_error("Embedding: index must be Int64");
    }

    size_t num_tokens = index->numel();
    size_t hidden_size = weight->shape().back();
    size_t vocab_size = weight->shape()[0];
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(), num_tokens, vocab_size, hidden_size, out->dtype());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::embedding(out->data(), index->data(), weight->data(), num_tokens, vocab_size, hidden_size, out->dtype());
#endif
    default:
        throw std::runtime_error("Embedding: device not supported");
    }
}
} // namespace llaisys::ops
