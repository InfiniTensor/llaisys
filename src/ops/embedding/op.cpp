#include "op.hpp"
#include "../../utils.hpp"
#include "cpu/embedding_cpu.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "Embedding: index tensor must be of type INT64.");
    // out shape: [index.shape[0], weight.shape[1]]
    ASSERT(out->shape().size() == 2, "Embedding: output tensor must be 2-dimensional.");
    ASSERT(index->shape().size() == 1, "Embedding: index tensor must be 1-dimensional.");
    ASSERT(weight->shape().size() == 2, "Embedding: weight tensor must be 2-dimensional.");
    ASSERT(out->shape()[0] == index->shape()[0], "Embedding: output tensor's first dimension must match index tensor's size.");
    ASSERT(out->shape()[1] == weight->shape()[1], "Embedding: output tensor's second dimension must match weight tensor's second dimension.");
    ASSERT(weight->isContiguous() && out->isContiguous() && index->isContiguous(), "Embedding: weight index and weight tensors must be contiguous.");

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), weight->data(), out->dtype(), index->shape()[0], weight->shape()[0], weight->shape()[1]);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(), out->dtype(), index->shape()[0], weight->shape()[0], weight->shape()[1]);
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
