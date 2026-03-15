#include "op.hpp"


#include "./cpu/rms_norm_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "./nvidia/rms_norm_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "./metax/rms_norm_metax.hpp"
#endif
#include "llaisys.h"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    // 1. 参数校验
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_ARGUMENT(out->ndim() == 2, "out must be 2d");
    CHECK_ARGUMENT(in->ndim() == 2, "in must be 2d");
    CHECK_ARGUMENT(weight->ndim() == 1, "weight must be 1d");
    CHECK_ARGUMENT(out->shape()[0] == in->shape()[0] && out->shape()[1] == in->shape()[1], 
                   "out's shape must be same as in's shape");
    CHECK_ARGUMENT(weight->shape()[0] == out->shape()[1], 
                   "weight and out must have equal N");
    CHECK_ARGUMENT(out->dtype() == in->dtype() && out->dtype() == weight->dtype(),
                   "tensors must have the same dtype");
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), 
            "tensors must be contiguous");

    // 2. 设置设备上下文
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    // 3. 张量分发到指定设备
    size_t M = out->shape()[0];
    size_t N = out->shape()[1];

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(),
                             in->data(),
                             weight->data(),
                             out->dtype(),
                             M, N, eps);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rms_norm(out->data(), in->data(), weight->data(),
                               out->dtype(), M, N, eps);
#endif
#ifdef ENABLE_METAX_API
    case LLAISYS_DEVICE_METAX:
        return metax::rms_norm(out->data(), in->data(), weight->data(),
                               out->dtype(), M, N, eps);
#endif
        default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
