#include "op.hpp"

#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_SAME_DTYPE(pos_ids->dtype(), LLAISYS_DTYPE_I64);
    ASSERT(in->ndim() == 3, "Rope: input tensor must be 3-dimensional.");
    ASSERT(out->ndim() == 3, "Rope: output tensor must be 3-dimensional.");
    ASSERT(pos_ids->ndim() == 1, "Rope: position ids tensor must be 1-dimensional.");
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    ASSERT(in->shape()[0] == pos_ids->shape()[0], "Rope: position ids length must match input tensor sequence length.");
    ASSERT(in->shape()[2] % 2 == 0, "Rope: head dimension must be even.");
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(), "Rope: all tensors must be contiguous.");

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), pos_ids->data(), out->dtype(), in->shape()[0], in->shape()[1], in->shape()[2], theta);
    }
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(), out->dtype(), in->shape()[0], in->shape()[1], in->shape()[2], theta);
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
