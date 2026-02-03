#include "op.hpp"
#include "../../utils.hpp"
#include "cpu/swiglu_cpu.hpp"

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());

    ASSERT(out->ndim() == 2 && gate->ndim() == 2 && up->ndim() == 2, "Tensors must be 2D");
    ASSERT(out->shape() == gate->shape() && out->shape() == up->shape(), "Shapes must match");
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(), "Tensors must be contiguous");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        cpu::swiglu(out->data(), gate->data(), up->data(), out->dtype(), out->numel());
        return;
    }
    
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
