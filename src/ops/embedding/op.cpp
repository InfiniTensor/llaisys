#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"

namespace llaisys::ops {

void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    
    CHECK_SAME_DEVICE(out, index, weight);

    
    ASSERT(index->ndim() == 1, "embedding: index must be 1D");
    ASSERT(weight->ndim() == 2, "embedding: weight must be 2D");
    ASSERT(out->ndim() == 2, "embedding: out must be 2D");

    const size_t N = index->shape()[0];
    const size_t V = weight->shape()[0];
    const size_t D = weight->shape()[1];

    ASSERT(out->shape()[0] == N, "embedding: out.shape[0] must equal index.shape[0]");
    ASSERT(out->shape()[1] == D, "embedding: out.shape[1] must equal weight.shape[1]");

    
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "embedding: index must be int64");
    ASSERT(out->dtype() == weight->dtype(), "embedding: out dtype must match weight dtype");

    
    ASSERT(out->isContiguous(), "embedding: out must be contiguous");
    ASSERT(weight->isContiguous(), "embedding: weight must be contiguous");
    ASSERT(index->isContiguous(), "embedding: index must be contiguous");

    
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(),
                              reinterpret_cast<const int64_t *>(index->data()),
                              weight->data(),
                              out->dtype(),
                              N, D, V);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(),
                              reinterpret_cast<const int64_t *>(index->data()),
                              weight->data(),
                              out->dtype(),
                              N, D, V);
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
