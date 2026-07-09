#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "cuda/embedding_cuda.cuh"
#endif

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "Embedding: index must be int64.");
    ASSERT(weight->ndim() == 2, "Embedding: weight must be 2D.");
    ASSERT(out->ndim() == 2, "Embedding: out must be 2D.");
    ASSERT(out->isContiguous() && weight->isContiguous(), "Embedding: tensors must be contiguous.");

    size_t n_idx = index->numel();
    size_t embd_dim = weight->shape()[1];

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), weight->data(), weight->dtype(), n_idx, embd_dim);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(), weight->dtype(), n_idx, embd_dim);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return cuda::embedding(out->data(), index->data(), weight->data(), weight->dtype(), n_idx, embd_dim);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
