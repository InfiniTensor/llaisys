#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "cuda/linear_cuda.cuh"
#endif

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    ASSERT(out->ndim() == 2 && in->ndim() == 2 && weight->ndim() == 2,
           "Linear: out, in, weight must be 2D.");
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "Linear: tensors must be contiguous.");

    size_t M = in->shape()[0];
    size_t K = in->shape()[1];
    size_t N = weight->shape()[0];

    bool has_bias = (bias != nullptr);
    const std::byte *bias_data = has_bias ? bias->data() : nullptr;

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(), bias_data,
                           out->dtype(), M, N, K, has_bias);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(), bias_data,
                           out->dtype(), M, N, K, has_bias);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return cuda::linear(out->data(), in->data(), weight->data(), bias_data,
                            out->dtype(), M, N, K, has_bias);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
