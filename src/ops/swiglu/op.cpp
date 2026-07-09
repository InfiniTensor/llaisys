#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/swiglu_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/swiglu_nvidia.hpp"
#endif

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    // 基础检查
    CHECK_SAME_DEVICE(out, gate, up);
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());

    // 形状检查
    ASSERT(out->shape() == gate->shape() && out->shape() == up->shape(), 
           "SwigLU: out, gate, and up must have the same shape.");
    
    // 连续性检查
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(), 
           "SwigLU: all tensors must be contiguous.");

    size_t num_elements = out->numel();

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::swiglu(out->data(), gate->data(), up->data(), 
                           out->dtype(), num_elements);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::swiglu(out->data(), gate->data(), up->data(), out->dtype(), num_elements);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::swiglu(out->data(), gate->data(), up->data(), out->dtype(), num_elements);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
}
