#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/rms_norm_cuda.cuh"
#endif

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    // 检查设备一致性
    CHECK_SAME_DEVICE(out, in, weight);

    // 检查数据类型一致性
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());

    // 检查维度
    CHECK_ARGUMENT(in->ndim() == 2, "rms_norm: input must be 2D tensor");
    CHECK_ARGUMENT(out->ndim() == 2, "rms_norm: output must be 2D tensor");
    CHECK_ARGUMENT(weight->ndim() == 1, "rms_norm: weight must be 1D tensor");

    // 检查形状兼容性
    size_t num_rows = in->shape()[0];
    size_t row_dim = in->shape()[1];

    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_ARGUMENT(weight->shape()[0] == row_dim,
                   "rms_norm: weight dimension must match input's row dimension");

    // 检查所有张量都是连续的
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "rms_norm: all tensors must be contiguous.");

    // 设置设备上下文
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), weight->data(),
                             out->dtype(), num_rows, row_dim, eps);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rms_norm(out->data(), in->data(), weight->data(),
                                out->dtype(), num_rows, row_dim, eps);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
