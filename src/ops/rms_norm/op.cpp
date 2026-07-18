#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/rms_norm_nvidia.cuh"
#elif defined(ENABLE_METAX_API)
#include "metax/rms_norm_metax.cuh"
#endif

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "RMS Norm: inputs must be contiguous.");
    ASSERT(in->ndim() == 2, "RMS Norm: input tensor must be 2D.");
    ASSERT(out->ndim() == 2, "RMS Norm: output tensor must be 2D.");
    ASSERT(weight->ndim() == 1, "RMS Norm: weight tensor must be 1D.");

    size_t N = in->shape()[0];
    size_t D = in->shape()[1];
    ASSERT(out->shape() == in->shape(), "RMS Norm: output shape must match input shape.");
    ASSERT(weight->shape()[0] == D, "RMS Norm: weight shape must match input hidden size.");

    // always support cpu calculation
    if (in->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out->data(), in->data(), weight->data(), eps, in->dtype(), N, D);
    }

    llaisys::core::context().setDevice(in->deviceType(), in->deviceId());

    switch (in->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), weight->data(), eps, in->dtype(), N, D);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rms_norm(out->data(), in->data(), weight->data(), eps, in->dtype(), N, D);
#elif defined(ENABLE_METAX_API)
    case LLAISYS_DEVICE_METAX:
        return metax::rms_norm(out->data(), in->data(), weight->data(), eps, in->dtype(), N, D);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
