#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/linear_nvidia.cuh"
#elif defined(ENABLE_METAX_API)
#include "metax/linear_metax.cuh"
#endif

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    if (bias) CHECK_SAME_DEVICE(out, bias);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    CHECK_SAME_DTYPE(in->dtype(), weight->dtype());
    if (bias) CHECK_SAME_DTYPE(in->dtype(), bias->dtype());

    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "Linear: output, input tensor and weight tensor must be contiguous.");
    ASSERT(in->ndim() == 2, "Linear: input tensor must be 2D.");
    ASSERT(weight->ndim() == 2, "Linear: weight tensor must be 2D.");
    ASSERT(out->ndim() == 2, "Linear: output tensor must be 2D.");
    ASSERT(in->shape()[1] == weight->shape()[1], "Linear: input and weight shapes are incompatible.");
    ASSERT(out->shape()[0] == in->shape()[0], "Linear: out.shape[0] must match input batch size.");
    ASSERT(out->shape()[1] == weight->shape()[0], "Linear: out.shape[1] must match weight.shape[0].");
    if (bias) {
        ASSERT(bias->isContiguous(), "Linear: bias tensor must be contiguous.");
        ASSERT(bias->ndim() == 1, "Linear: bias tensor must be 1D.");
        ASSERT(bias->shape()[0] == weight->shape()[0], "Linear: bias shape must match weight.shape[0].");
    }

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(), bias ? bias->data() : nullptr, in->dtype(), in->shape()[0], in->shape()[1], weight->shape()[0]);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(), bias ? bias->data() : nullptr, in->dtype(), in->shape()[0], in->shape()[1], weight->shape()[0]);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::linear(out->data(), in->data(), weight->data(), bias ? bias->data() : nullptr,
                              in->dtype(), in->shape()[0], in->shape()[1], weight->shape()[0]);
#elif defined(ENABLE_METAX_API)
    case LLAISYS_DEVICE_METAX:
        return metax::linear(out->data(), in->data(), weight->data(), bias ? bias->data() : nullptr,
                             in->dtype(), in->shape()[0], in->shape()[1], weight->shape()[0]);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
