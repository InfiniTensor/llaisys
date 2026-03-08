#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include <vector>

#include "cpu/add_cpu.hpp"

namespace llaisys::ops {
void add(tensor_t c, tensor_t a, tensor_t b) {
    CHECK_SAME_DEVICE(c, a, b);
    // Only support contiguous inputs with same shape for now.
    CHECK_SAME_SHAPE(c->shape(), a->shape(), b->shape());
    CHECK_SAME_DTYPE(c->dtype(), a->dtype(), b->dtype());
    ASSERT(c->isContiguous() && a->isContiguous() && b->isContiguous(), "Add: all tensors must be contiguous.");

    // always support cpu calculation
    if (c->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::add(c->data(), a->data(), b->data(), c->dtype(), c->numel());
    }

    // set device context
    llaisys::core::context().setDevice(c->deviceType(), c->deviceId());

    switch (c->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::add(c->data(), a->data(), b->data(), c->dtype(), c->numel());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA: {
        auto &ctx = llaisys::core::context();
        ctx.setDevice(c->deviceType(), c->deviceId());
        const auto *api = ctx.runtime().api();

        const size_t bytes = c->numel() * c->elementSize();
        std::vector<std::byte> hc(bytes), ha(bytes), hb(bytes);
        api->memcpy_sync(ha.data(), a->data(), bytes, LLAISYS_MEMCPY_D2H);
        api->memcpy_sync(hb.data(), b->data(), bytes, LLAISYS_MEMCPY_D2H);

        cpu::add(hc.data(), ha.data(), hb.data(), c->dtype(), c->numel());

        api->memcpy_sync(c->data(), hc.data(), bytes, LLAISYS_MEMCPY_H2D);
        return;
    }
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
