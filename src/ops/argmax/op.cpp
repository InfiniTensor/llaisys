#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/argmax_cpu.hpp"

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    // 1. 基础校验
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    CHECK_SAME_DTYPE(vals->dtype(), max_val->dtype());
    
    // 2. 形状校验 (根据需求：vals 为 1D，输出为标量)
    CHECK_ARGUMENT(vals->ndim() == 1, "Argmax input 'vals' must be 1D tensor.");
    CHECK_ARGUMENT(max_idx->numel() == 1, "Argmax output 'max_idx' must be scalar (1 element).");
    CHECK_ARGUMENT(max_val->numel() == 1, "Argmax output 'max_val' must be scalar (1 element).");
    CHECK_ARGUMENT(vals->numel() > 0, "Argmax input 'vals' cannot be empty.");

    // 3. 内存连续性校验
    ASSERT(vals->isContiguous() && max_idx->isContiguous() && max_val->isContiguous(), 
           "Argmax: all tensors must be contiguous.");

    // 4. CPU 分发
    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::argmax(vals->data(), max_val->data(), max_idx->data(),
                           vals->dtype(), max_idx->dtype(), vals->numel());
    }

    // 5. 设备上下文设置 (为 GPU 准备)
    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());

    // 目前仅支持 CPU，GPU 分支待实现
    TO_BE_IMPLEMENTED();
}
} // namespace llaisys::ops
