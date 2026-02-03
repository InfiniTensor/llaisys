#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/rearrange_cpu.hpp"


namespace llaisys::ops {
void rearrange(tensor_t out, tensor_t in) {
    CHECK_SAME_DEVICE(out, in);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_SAME_SHAPE(out->shape(), in->shape());

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        cpu::rearrange(out->data(), in->data(), out->shape(), out->strides(), in->strides(), out->dtype());
        return;
    }
}
} // namespace llaisys::ops
