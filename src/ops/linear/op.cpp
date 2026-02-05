#include "op.hpp"

#include "../../utils.hpp"

#include "cpu/linear.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight, bias);
    CHECK_ARGUMENT(
        in->shape().back() == weight->shape().back(),
        "Input dimension does not match weight dimension."
    );

    size_t batch_size = 1;
    for (size_t i = 0; i < in->ndim() - 1; ++i) {
        batch_size *= in->shape()[i];
    }
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(
            out->data(),
            in->data(),
            weight->data(),
            bias->data(),
            batch_size,
            in->shape().back(),
            weight->shape()[weight->ndim() - 2],
            in->dtype()
        );
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

void linear(tensor_t out, tensor_t in, tensor_t weight) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_ARGUMENT(
        in->shape().back() == weight->shape()[weight->ndim() - 2],
        "Input dimension does not match weight dimension."
    );

    size_t batch_size = 1;
    for (size_t i = 0; i < in->ndim() - 1; ++i) {
        batch_size *= in->shape()[i];
    }
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(
            out->data(),
            in->data(),
            weight->data(),
            nullptr,
            batch_size,
            in->shape().back(),
            weight->shape().back(),
            in->dtype()
        );
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
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
