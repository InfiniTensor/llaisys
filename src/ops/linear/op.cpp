#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    // out = in @ weight^T + bias
    // in: (batch, in_features)
    // weight: (out_features, in_features)
    // out: (batch, out_features)
    // bias: (out_features,) or nullptr

    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());

    ASSERT(in->ndim() == 2, "input must be 2D");
    ASSERT(weight->ndim() == 2, "weight must be 2D");
    ASSERT(out->ndim() == 2, "output must be 2D");

    size_t batch = in->shape()[0];
    size_t in_features = in->shape()[1];
    size_t out_features = weight->shape()[0];

    ASSERT(weight->shape()[1] == in_features, "weight shape mismatch");
    ASSERT(out->shape()[0] == batch && out->shape()[1] == out_features, "output shape mismatch");

    bool has_bias = (bias != nullptr);
    if (has_bias) {
        CHECK_SAME_DEVICE(out, bias);
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
        ASSERT(bias->ndim() == 1 && bias->shape()[0] == out_features, "bias shape mismatch");
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(),
                          has_bias ? bias->data() : nullptr,
                          out->dtype(), batch, in_features, out_features, has_bias);
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
