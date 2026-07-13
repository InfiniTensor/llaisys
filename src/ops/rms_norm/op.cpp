#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/rms_norm_nvidia.hpp"
#endif

namespace llaisys::ops {
void rms_norm(tensor_t c, tensor_t a, tensor_t b, float eps) {
    CHECK_SAME_DEVICE(c, a, b);
    CHECK_SAME_DTYPE(c->dtype(), a->dtype(), b->dtype());
    ASSERT(c->isContiguous() && a->isContiguous() && b->isContiguous(), "RMSNorm: all tensors must be contiguous.");
    
    // 计算特征维度 (dim) 和 Token 总数 (rows)
    size_t dim = a->shape().back();
    size_t rows = a->numel() / dim;

    if (c->deviceType() == LLAISYS_DEVICE_CPU) {
        // 修复：将 c->dtype() 移到第 4 个参数位置，对齐 CPU 版本的签名
        return cpu::rms_norm(c->data(), a->data(), b->data(), c->dtype(), rows, dim, eps);
    }

    llaisys::core::context().setDevice(c->deviceType(), c->deviceId());

    switch (c->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(c->data(), a->data(), b->data(), c->dtype(), rows, dim, eps);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        // NVIDIA 版本按照我们刚写的头文件签名，type 在最后
        return nvidia::rms_norm(c->data(), a->data(), b->data(), rows, dim, eps, c->dtype());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops