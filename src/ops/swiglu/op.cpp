#include "op.hpp"
#include "cpu/swiglu.hpp"

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);
    CHECK_SAME_SHAPE(out->shape(), gate->shape(), up->shape());

   if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::swiglu(
            out->data(),
            up->data(),
            gate->data(),
            out->dtype(),
            out->numel()
        );
    } else {
        TO_BE_IMPLEMENTED();
   }
}
} // namespace llaisys::ops
