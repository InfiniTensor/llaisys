#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "./cpu/argmax_cpu.hpp"

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    // 设备一致性：确保输入张量vals和输出张量max_idx、max_val都在同一个设备上
    // （例如都在 CPU 上，或都在同一块 GPU 上）。跨设备操作在算子内部是不允许的。
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    // 类型一致性：要求存储最大值的张量 max_val 必须与输入张量 vals 具有相同的数据类型
    CHECK_SAME_DTYPE(vals->dtype(), max_val->dtype());
    
    // 2. 形状校验 (根据需求：vals 为 1D，输出为标量)
    // 标量输出：因为是对整个 1D 向量求最大值，所以输出的索引和数值都必须是只有一个元素的标量。
    CHECK_ARGUMENT(vals->ndim() == 1, "Argmax input 'vals' must be 1D tensor.");
    CHECK_ARGUMENT(max_idx->numel() == 1, "Argmax output 'max_idx' must be scalar (1 element).");
    CHECK_ARGUMENT(max_val->numel() == 1, "Argmax output 'max_val' must be scalar (1 element).");
    CHECK_ARGUMENT(vals->numel() > 0, "Argmax input 'vals' cannot be empty.");

    // 3. 内存连续性校验
    // 连续性要求：底层 CPU 内核（argmax_cpu.cpp）通常假设数据在内存中是连续排列的，以便使用简单的循环遍历。
    // 如果张量经过了 slice 或 permute 操作导致内存不连续，这里会触发断言失败。
    ASSERT(vals->isContiguous() && max_idx->isContiguous() && max_val->isContiguous(), 
           "Argmax: all tensors must be contiguous.");

    // 4. CPU 分发
    // 意它传递的是 原始指针 (data())。这对应了你在 argmax_cpu.cpp 中看到的实现，
    // 那里会根据 dtype 使用 switch-case 进一步分发到具体的模板函数 argmax_kernel
    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::argmax(vals->data(), max_val->data(), max_idx->data(),
                           vals->dtype(), max_idx->dtype(), vals->numel());
    }

    // 5. 设备上下文设置(为GPU准备)
    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());

    // 目前仅支持 CPU，GPU 分支待实现
    TO_BE_IMPLEMENTED();
}
} // namespace llaisys::ops