#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_ARGUMENT(out->ndim() == 2, "out must be a 2D tensor");
    CHECK_ARGUMENT(index->ndim() == 1, "index must be a 1D tensor");
    CHECK_ARGUMENT(weight->ndim() == 2, "weight must be a 2D tensor");
    CHECK_ARGUMENT(index->dtype() == LLAISYS_DTYPE_I64, "index must have dtype I64");
    CHECK_ARGUMENT(out->shape()[0] == index->shape()[0], "out must have the same shape as index");
    CHECK_ARGUMENT(out->shape()[1] == weight->shape()[1], "out must have the same shape as weight");
    CHECK_ARGUMENT(out->isContiguous(), "embedding: out tensor must be contiguous.");
    CHECK_ARGUMENT(index->isContiguous(), "embedding: index tensor must be contiguous.");
    CHECK_ARGUMENT(weight->isContiguous(), "embedding: weight tensor must be contiguous.");

    // 总是支持CPU计算
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), weight->data(), out->dtype(), out->numel(), weight->shape()[1]);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(), out->dtype(), out->numel(), weight->shape()[1]);
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
