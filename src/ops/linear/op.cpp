#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/linear_cuda.cuh"
#endif

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    // 检查设备一致性
    CHECK_SAME_DEVICE(out, in, weight);
    if (bias) {
        CHECK_SAME_DEVICE(out, bias);
    }

    // 检查数据类型一致性
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    if (bias) {
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
    }

    // 检查维度
    CHECK_ARGUMENT(in->ndim() == 2, "linear: input must be 2D tensor");
    CHECK_ARGUMENT(weight->ndim() == 2, "linear: weight must be 2D tensor");
    CHECK_ARGUMENT(out->ndim() == 2, "linear: output must be 2D tensor");
    if (bias) {
        CHECK_ARGUMENT(bias->ndim() == 1, "linear: bias must be 1D tensor");
    }

    // 获取形状参数
    size_t batch_size = in->shape()[0];
    size_t in_features = in->shape()[1];
    size_t out_features = weight->shape()[0];

    CHECK_ARGUMENT(weight->shape()[1] == in_features,
                   "linear: weight's second dimension must match input's second dimension");
    CHECK_ARGUMENT(out->shape()[0] == batch_size,
                   "linear: output's first dimension must match input's first dimension");
    CHECK_ARGUMENT(out->shape()[1] == out_features,
                   "linear: output's second dimension must match weight's first dimension");
    if (bias) {
        CHECK_ARGUMENT(bias->shape()[0] == out_features,
                       "linear: bias dimension must match weight's first dimension");
    }

    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "linear: all tensors must be contiguous.");
    if (bias) {
        ASSERT(bias->isContiguous(), "linear: bias must be contiguous.");
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(),
                           bias ? bias->data() : nullptr,
                           out->dtype(), batch_size, in_features, out_features);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::linear(out->data(), in->data(), weight->data(),
                              bias ? bias->data() : nullptr,
                              out->dtype(), batch_size, in_features, out_features);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
