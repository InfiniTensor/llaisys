#include "op.hpp"
#include "../../utils.hpp"
#include "./cpu/rearrange_cpu.hpp"

namespace llaisys::ops {
void rearrange(tensor_t out, tensor_t in) {
    // 1. 基础校验
    CHECK_SAME_DEVICE(out, in);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_SAME_SHAPE(out->shape(), in->shape());

    // 2. 设备分发
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        cpu::rearrange((std::byte *)out->data(), (const std::byte *)in->data(), out->shape(), out->strides(), in->strides(), out->dtype());
        return;
    }

    TO_BE_IMPLEMENTED();
}
} // namespace llaisys::ops
