#include "op.hpp"
#include "cpu/argmax_cpu.hpp"
#include "../../core/context/context.hpp"
#include "../../utils.hpp"
#include <stdexcept>

namespace llaisys::ops::nvidia {
#ifdef ENABLE_NVIDIA_API
    void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals);
#endif
}

namespace llaisys::ops {

void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    // 基础检查
    CHECK_SAME_DEVICE(max_idx, vals);
    
    // max_val 是可选的，只有非空时才检查设备
    if (max_val) {
        CHECK_SAME_DEVICE(max_val, vals);
        //检查连续性
        ASSERT(max_val->isContiguous(), "Argmax max_val output must be contiguous");
    }

    ASSERT(max_idx->dtype() == LLAISYS_DTYPE_I64, "Argmax index output must be int32");
    ASSERT(vals->isContiguous() && max_idx->isContiguous(), "Argmax inputs/outputs must be contiguous");

    // 获取当前上下文应当运行的设备类型
    auto device_type = vals->deviceType();

    // 切换设备上下文
    llaisys::core::context().setDevice(device_type, vals->deviceId());

    switch (device_type) {
    case LLAISYS_DEVICE_CPU: {
        // 防止 max_val 为空时调用 ->data() 导致崩溃
        std::byte* val_ptr = max_val ? max_val->data() : nullptr;
        cpu::argmax(max_idx->data(), val_ptr, vals->data(), vals->dtype(), vals->numel());
        return;
    }

#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return llaisys::ops::nvidia::argmax(max_idx, max_val, vals);
#endif

    default:
        throw std::runtime_error("Argmax: Unsupported device type");
    }
}

} // namespace llaisys::ops