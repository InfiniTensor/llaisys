#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {

void linear(tensor_t out, tensor_t input, tensor_t weight, tensor_t bias) {

    bool has_bias = (bias != nullptr && bias->numel() > 0);

    if (has_bias) {
        CHECK_SAME_DEVICE(out, input, weight, bias);
        CHECK_SAME_DTYPE(out->dtype(), input->dtype(), weight->dtype(), bias->dtype());
        ASSERT(bias->isContiguous(), "Linear: bias must be contiguous.");
        ASSERT(bias->ndim() == 1, "Linear: bias must be 1D.");
    } else {
        CHECK_SAME_DEVICE(out, input, weight);
        CHECK_SAME_DTYPE(out->dtype(), input->dtype(), weight->dtype());
    }

    ASSERT(out->isContiguous() && input->isContiguous() && weight->isContiguous(),
           "Linear: tensors must be contiguous.");

    ASSERT(input->ndim() == 2, "Linear: input must be 2D [M, K].");
    ASSERT(weight->ndim() == 2, "Linear: weight must be 2D [N, K].");
    ASSERT(out->ndim() == 2, "Linear: output must be 2D [M, N].");

    size_t M = input->shape()[0];
    size_t K = input->shape()[1];
    size_t N = weight->shape()[0];

    // 检查矩阵乘法维度约束
    ASSERT(weight->shape()[1] == K,
           "Linear: weight input features (dim 1) must match input features (dim 1).");

    // 检查输出形状
    ASSERT(out->shape()[0] == M, "Linear: output dim 0 must match M.");
    ASSERT(out->shape()[1] == N, "Linear: output dim 1 must match N.");

    // 检查 bias 维度
    if (has_bias) {
        ASSERT(bias->shape()[0] == N, "Linear: bias size must match output features N.");
    }

    const std::byte *bias_ptr = has_bias ? bias->data() : nullptr;

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        ::llaisys::ops::cpu::linear(
            input->data(),
            weight->data(),
            bias_ptr,
            out->data(),
            out->dtype(),
            M, K, N);
        return;
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        ::llaisys::ops::cpu::linear(
            input->data(), weight->data(), bias_ptr, out->data(),
            out->dtype(), M, K, N);
        return;

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
