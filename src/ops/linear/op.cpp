#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_ARGUMENT(out->ndim() == 2, "out must be a 2D tensor");
    CHECK_ARGUMENT(in->ndim() == 2, "in must be a 2D tensor");
    CHECK_ARGUMENT(weight->ndim() == 2, "weight must be a 2D tensor");
    CHECK_ARGUMENT(out->shape()[0] == in->shape()[0], "out and in must have the same batch size");
    CHECK_ARGUMENT(out->shape()[1] == weight->shape()[0], "out features must match weight rows");
    CHECK_ARGUMENT(in->shape()[1] == weight->shape()[1], "in features must match weight columns");
    if (bias != nullptr) {
        CHECK_ARGUMENT(bias->ndim() == 1, "bias must be a 1D tensor");
        CHECK_ARGUMENT(bias->shape()[0] == out->shape()[1], "bias size must match out features");
    }
    CHECK_ARGUMENT(out->isContiguous(), "linear: out tensor must be contiguous.");
    CHECK_ARGUMENT(in->isContiguous(), "linear: in tensor must be contiguous.");
    CHECK_ARGUMENT(weight->isContiguous(), "linear: weight tensor must be contiguous.");
    if (bias != nullptr) {
        CHECK_ARGUMENT(bias->isContiguous(), "linear: bias tensor must be contiguous.");
    }

    size_t batch_size = in->shape()[0];
    size_t in_features = in->shape()[1];
    size_t out_features = out->shape()[1];

    // 总是支持CPU计算
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(
            out->data(), 
            in->data(), 
            weight->data(), 
            bias ? bias->data() : nullptr, 
            out->dtype(), 
            batch_size, 
            in_features, 
            out_features
        );
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(
            out->data(), 
            in->data(), 
            weight->data(), 
            bias ? bias->data() : nullptr, 
            out->dtype(), 
            batch_size, 
            in_features, 
            out_features
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
