#include "op.hpp"
#include "cpu/rms_norm_cpu.hpp"
#include "../../core/context/context.hpp"
#include "../../utils.hpp"
#include <stdexcept>

namespace llaisys::ops::nvidia {
#ifdef ENABLE_NVIDIA_API
    void rmsnorm(tensor_t out, tensor_t in, tensor_t weight, float eps);
#endif
}

namespace llaisys::ops {

void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    // 1. 物理设备与介质隔离校验
    CHECK_SAME_DEVICE(out, in);
    CHECK_SAME_DEVICE(out, weight);

    auto device = core::context().runtime().deviceType();

    // 2. 硬件上下文强制切换，防止多卡显存踩踏
    llaisys::core::context().setDevice(device, out->deviceId());

    // 3. 算子路由与分发
    if (device == LLAISYS_DEVICE_CPU) {
        
        // 提取 Hidden_dim，将前端多维逻辑张量展平为底层处理所需的二维矩阵
        size_t cols = in->shape().back();
        size_t rows = in->numel() / cols;

        if (weight->numel() != cols) {
             throw std::invalid_argument("RMSNorm: Weight shape mismatch. Must match Hidden_dim.");
        }

        cpu::rms_norm(out->data(), in->data(), weight->data(),
                      in->dtype(), rows, cols, eps);

    } 
#ifdef ENABLE_NVIDIA_API
    else if (device == LLAISYS_DEVICE_NVIDIA) {
        llaisys::ops::nvidia::rmsnorm(out, in, weight, eps);
    }
#endif
    else {
        throw std::runtime_error("RMSNorm: Unsupported device type");
    }
}

} // namespace llaisys::ops