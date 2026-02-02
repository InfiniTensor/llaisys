#include "op.hpp"
#include "../../utils.hpp"
#include "./cpu/swiglu_cpu.hpp"

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    // 1. 基础校验
    CHECK_SAME_DEVICE(out, gate, up);
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());
    CHECK_SAME_SHAPE(out->shape(), gate->shape());
    CHECK_SAME_SHAPE(out->shape(), up->shape());
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(), "SwiGLU: tensors must be contiguous");

    // 2. 设备分发
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        cpu::swiglu(out->data(), gate->data(), up->data(), out->dtype(), out->numel());
        return;
    }

    TO_BE_IMPLEMENTED();
}
} // namespace llaisys::ops
