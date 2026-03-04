#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/swiglu_cpu.hpp"


#ifdef ENABLE_NVIDIA_API
#include "nvidia/swiglu_nvidia.hpp"
#endif

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());

    ASSERT(out->ndim() == 2 && gate->ndim() == 2 && up->ndim() == 2, "SwiGLU: tensors must be 2D.");
    ASSERT(out->shape() == gate->shape() && out->shape() == up->shape(), "SwiGLU: shapes must match.");
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(), "SwiGLU: tensors must be contiguous.");

    size_t numel = out->numel();

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::swiglu(out->data(), gate->data(), up->data(), out->dtype(), numel);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::swiglu(out->data(), gate->data(), up->data(), out->dtype(), numel);
#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        return nvidia::swiglu(out, gate, up);
    }
#endif

    if (dtype == LLAISYS_DTYPE_F32) {
        swiglu_cpu_kernel<float>(out, gate, up);
    } 
    else if (dtype == LLAISYS_DTYPE_F16) { 
        swiglu_cpu_kernel<llaisys::fp16_t>(out, gate, up);
    } 
    else if (dtype == LLAISYS_DTYPE_BF16) { 
        swiglu_cpu_kernel<llaisys::bf16_t>(out, gate, up);
    }
    else {
        throw std::runtime_error("SwiGLU: Unsupported dtype");
    }
}
} // namespace llaisys::ops
