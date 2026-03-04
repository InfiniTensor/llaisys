#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/swiglu_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/swiglu_nvidia.hpp"
#endif

namespace llaisys::ops {

// 执行 SwiGLU 激活函数：output = silu(gate) * up
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    // 检查所有张量是否在同一设备
    CHECK_SAME_DEVICE(out, gate, up);
    // 数据类型必须完全一致
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());

    // 所有输入和输出必须是二维张量 [batch_size, hidden_dim]
    ASSERT(out->ndim() == 2 && gate->ndim() == 2 && up->ndim() == 2, 
           "SwiGLU: tensors must be 2-dimensional.");
    
    // 验证形状完全匹配
    ASSERT(out->shape() == gate->shape() && out->shape() == up->shape(), 
           "SwiGLU: output, gate, and up shapes must match exactly.");

    // 所有张量必须内存连续
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(), 
           "SwiGLU: all tensors must be contiguous in memory.");

    size_t numel = out->numel();

    // CPU 路径直接返回
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::swiglu(
            out->data(),
            gate->data(),
            up->data(),
            out->dtype(),
            numel
        );
    }

    // 设置当前设备上下文
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        // 理论上不会执行到这里，因为上面已经处理了 CPU 情况
        return cpu::swiglu(
            out->data(),
            gate->data(),
            up->data(),
            out->dtype(),
            numel
        );
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        nvidia::swiglu(
            out->data(),
            gate->data(),
            up->data(),
            out->dtype(),
            numel
        );
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace llaisys::ops