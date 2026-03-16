/**
 * Add 算子实现：校验输入合法性，按设备分发到具体实现（当前仅 CPU）。
 *
 * 被 src/llaisys/ops.cc 的 llaisysAdd 调用，供 Python 与 qwen2 等 C++ 代码使用。
 */
#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/add_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "llaisys/ops_nvidia.h"
#endif

namespace llaisys::ops {

void add(tensor_t c, tensor_t a, tensor_t b) {
    // ---------- 合法性检查：同设备、同形状、同 dtype、三者皆连续 ----------
    CHECK_SAME_DEVICE(c, a, b);
    CHECK_SAME_SHAPE(c->shape(), a->shape(), b->shape());
    CHECK_SAME_DTYPE(c->dtype(), a->dtype(), b->dtype());
    ASSERT(c->isContiguous() && a->isContiguous() && b->isContiguous(), "Add: all tensors must be contiguous.");

    // CPU 分支：直接调 CPU 实现，无需切换 Context
    if (c->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::add(c->data(), a->data(), b->data(), c->dtype(), c->numel());
    }

    // 非 CPU 时先切到当前张量所在设备，再按设备类型分发
    llaisys::core::context().setDevice(c->deviceType(), c->deviceId());

    switch (c->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::add(c->data(), a->data(), b->data(), c->dtype(), c->numel());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        nvidia::add(c->data(), a->data(), b->data(), c->dtype(), c->numel());
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
