#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/argmax_cpu.hpp"

namespace llaisys::ops {

void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    // 参数检查
    CHECK_ARGUMENT(max_idx->ndim() == 1, "max_idx must be a 1D tensor");
    CHECK_ARGUMENT(max_val->ndim() == 1 && max_val->shape()[0] == 1, "max_val must be a 1D tensor with one element");
    CHECK_ARGUMENT(max_val->dtype() == vals->dtype(), "max_val must have the same dtype as vals");
    CHECK_ARGUMENT(max_idx->dtype() == LLAISYS_DTYPE_I64, "max_idx must have dtype I64");
    CHECK_ARGUMENT(vals->isContiguous(), "argmax: vals tensor must be contiguous.");

    // 总是支持CPU计算
    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
    }

    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());

    switch (vals->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
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