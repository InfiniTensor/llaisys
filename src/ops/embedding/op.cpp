#include "op.hpp"
#include "cpu/embedding_cpu.hpp"
#include "llaisys.h"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/embedding_cu.cuh"
#endif

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "embedding(): index.dtype must be int64");
    ASSERT(out->isContiguous(), "embedding(): output tensor must be contiguous");
    ASSERT(index->isContiguous(), "embedding(): index tensor must be contiguous");
    ASSERT(weight->isContiguous(), "embedding(): weight tensor must be contiguous");

    auto device = out->deviceType();
    core::context().setDevice(device, out->deviceId());

    auto embedding_dim = weight->shape().back();
    if (device == LLAISYS_DEVICE_CPU) {
        llaisys::ops::cpu::embedding(out->data(), index->data(), weight->data(), index->numel(), embedding_dim,
                                     out->dtype());
#ifdef ENABLE_NVIDIA_API
    } else if (device == LLAISYS_DEVICE_NVIDIA) {
        llaisys::ops::nvidia::embedding(out->data(), index->data(), weight->data(), index->numel(), embedding_dim,
                                        out->dtype());
#endif
    } else
        EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
