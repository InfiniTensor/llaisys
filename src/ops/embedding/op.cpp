#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/embedding_nvidia.cuh"
#endif

#ifdef ENABLE_METAX_API
#include "metax/embedding_metax.cuh"
#endif

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    // Only support 1D index and 2D weight for now
    ASSERT(index->ndim() == 1, "Embedding: index must be 1D tensor.");
    ASSERT(weight->ndim() == 2, "Embedding: weight must be 2D tensor.");
    ASSERT(out->ndim() == 2, "Embedding: out must be 2D tensor.");
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "Embedding: index must be int64 dtype.");
    ASSERT(out->dtype() == weight->dtype(), "Embedding: out and weight must have same dtype.");
    ASSERT(out->shape()[0] == index->numel(), "Embedding: out shape[0] must match index size.");
    ASSERT(out->shape()[1] == weight->shape()[1], "Embedding: out shape[1] must match weight shape[1].");
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(), "Embedding: all tensors must be contiguous.");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), weight->data(), weight->dtype(), index->numel(), weight->shape()[1]);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(), weight->dtype(), index->numel(), weight->shape()[1]);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::embedding(out->data(), index->data(), weight->data(), weight->dtype(), index->numel(), weight->shape()[1]);
#endif
#ifdef ENABLE_METAX_API
    case LLAISYS_DEVICE_METAX:
        return metax::embedding(out->data(), index->data(), weight->data(), weight->dtype(), index->numel(), weight->shape()[1]);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops