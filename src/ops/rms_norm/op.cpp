#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_ARGUMENT(out->ndim() == 2, "out must be a 2D tensor");
    CHECK_ARGUMENT(in->ndim() == 2, "in must be a 2D tensor");
    CHECK_ARGUMENT(weight->ndim() == 1, "weight must be a 1D tensor");
    CHECK_ARGUMENT(out->shape()[0] == in->shape()[0], "out and in must have the same batch size");
    CHECK_ARGUMENT(out->shape()[1] == in->shape()[1], "out and in must have the same hidden size");
    CHECK_ARGUMENT(weight->shape()[0] == in->shape()[1], "weight size must match hidden size");
    CHECK_ARGUMENT(out->isContiguous(), "rms_norm: out tensor must be contiguous.");
    CHECK_ARGUMENT(in->isContiguous(), "rms_norm: in tensor must be contiguous.");
    CHECK_ARGUMENT(weight->isContiguous(), "rms_norm: weight tensor must be contiguous.");

    size_t batch_size = in->shape()[0];
    size_t hidden_size = in->shape()[1];

    // 总是支持CPU计算
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(
            out->data(), 
            in->data(), 
            weight->data(), 
            eps, 
            out->dtype(), 
            batch_size, 
            hidden_size
        );
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(
            out->data(), 
            in->data(), 
            weight->data(), 
            eps, 
            out->dtype(), 
            batch_size, 
            hidden_size
        );
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
