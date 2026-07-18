#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/embedding_nvidia.cuh"
#elif defined(ENABLE_METAX_API)
#include "metax/embedding_metax.cuh"
#endif

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(),
           "Embedding: all tensors must be contiguous.");
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "Embedding: index tensor must be of type int64.");
    ASSERT(index->ndim() == 1, "Embedding: index tensor must be 1D.");
    ASSERT(weight->ndim() == 2, "Embedding: weight tensor must be 2D.");
    ASSERT(out->ndim() == 2, "Embedding: out tensor must be 2D.");
    ASSERT(out->shape()[0] == index->shape()[0], "Embedding: out.shape[0] must match index.shape[0].");
    ASSERT(out->shape()[1] == weight->shape()[1], "Embedding: out.shape[1] must match weight.shape[1].");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), weight->data(), weight->dtype(),
                              index->shape()[0], weight->shape()[1]);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(), weight->dtype(),
                              index->shape()[0], weight->shape()[1]);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::embedding(out->data(), index->data(), weight->data(), weight->dtype(),
                                 index->shape()[0], weight->shape()[1]);
#elif defined(ENABLE_METAX_API)
    case LLAISYS_DEVICE_METAX:
        return metax::embedding(out->data(), index->data(), weight->data(), weight->dtype(),
                                index->shape()[0], weight->shape()[1]);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
} // namespace llaisys::ops
}
