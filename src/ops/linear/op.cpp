#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/linear_nvidia.hpp"
#endif
#ifdef ENABLE_ILUVATAR_API
#include "nvidia/linear_nvidia.hpp"
#endif

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    if (bias) {
        CHECK_SAME_DEVICE(out, bias);
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
    }
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());

    ASSERT(out->ndim() == 2, "Linear: out must be 2D.");
    ASSERT(in->ndim() == 2, "Linear: input must be 2D.");
    ASSERT(weight->ndim() == 2, "Linear: weight must be 2D.");
    
    size_t m = in->shape()[0];
    size_t k = in->shape()[1];
    size_t n = weight->shape()[0]; // weight shape [out_features, in_features]

    ASSERT(weight->shape()[1] == k, "Linear: weight in_features mismatch.");
    ASSERT(out->shape()[0] == m && out->shape()[1] == n, "Linear: output shape mismatch.");
    if (bias) {
        ASSERT(bias->ndim() == 1 && bias->shape()[0] == n, "Linear: bias must be 1D with length out_features.");
    }

    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous()
               && (!bias || bias->isContiguous()),
           "Linear: all tensors must be contiguous.");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(), bias ? bias->data() : nullptr,
                           out->dtype(), m, n, k);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(), bias ? bias->data() : nullptr,
                           out->dtype(), m, n, k);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::linear(out->data(), in->data(), weight->data(), bias ? bias->data() : nullptr, out->dtype(),
                              m, n, k);
#endif
#ifdef ENABLE_ILUVATAR_API
    case LLAISYS_DEVICE_ILUVATAR:
        return nvidia::linear(out->data(), in->data(), weight->data(), bias ? bias->data() : nullptr, out->dtype(),
                              m, n, k);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
