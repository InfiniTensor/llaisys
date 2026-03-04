#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/rope_nvidia.hpp"
#endif

namespace llaisys::ops {

// 应用 Rotary Position Embedding (RoPE) 到输入张量
// 输入/输出形状：[seq_len, num_heads, head_dim]，要求 head_dim 为偶数
// position_ids: 形状为 [seq_len] 的 int64 张量，指定每个 token 的位置索引
void rope(tensor_t output, tensor_t input, tensor_t position_ids, float base_theta) {
    // 检查 output 和 input 是否在同一设备
    CHECK_SAME_DEVICE(output, input);
    // position_ids 也必须位于相同设备
    ASSERT(position_ids->deviceType() == output->deviceType() &&
           position_ids->deviceId() == output->deviceId(),
           "ROPE: position_ids must reside on the same device as input and output.");
    
    // 输入与输出数据类型必须一致
    CHECK_SAME_DTYPE(output->dtype(), input->dtype());
    // position_ids 必须为 int64 类型
    ASSERT(position_ids->dtype() == LLAISYS_DTYPE_I64,
           "ROPE: position_ids must be of type int64.");

    // 维度合法性检查
    ASSERT(output->ndim() == 3 && input->ndim() == 3,
           "ROPE: input and output tensors must be 3-dimensional with shape [seq_len, num_heads, head_dim].");
    ASSERT(position_ids->ndim() == 1,
           "ROPE: position_ids must be a 1-dimensional tensor of shape [seq_len].");

    size_t seq_len = input->shape()[0];
    size_t num_heads = input->shape()[1];
    size_t head_dim = input->shape()[2];
    
    // RoPE 要求每个头的维度为偶数
    ASSERT(head_dim % 2 == 0, "ROPE: head dimension must be even.");

    // 验证输出张量形状是否匹配输入
    ASSERT(output->shape()[0] == seq_len &&
           output->shape()[1] == num_heads &&
           output->shape()[2] == head_dim,
           "ROPE: output tensor shape does not match input shape.");
    
    // 验证 position_ids 长度与序列长度一致
    ASSERT(position_ids->shape()[0] == seq_len,
           "ROPE: length of position_ids must equal sequence length.");

    // 所有张量必须内存连续
    ASSERT(output->isContiguous() && input->isContiguous() && position_ids->isContiguous(),
           "ROPE: all tensors must be contiguous in memory.");

    // CPU 路径直接返回
    if (output->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(
            output->data(),
            input->data(),
            position_ids->data(),
            output->dtype(),
            seq_len,
            num_heads,
            head_dim,
            base_theta
        );
    }

    // 设置当前 CUDA 设备
    llaisys::core::context().setDevice(output->deviceType(), output->deviceId());

    switch (output->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        // 理论上不会执行到这里，因为上面已经处理了 CPU 情况，但为了完整性保留
        return cpu::rope(
            output->data(),
            input->data(),
            position_ids->data(),
            output->dtype(),
            seq_len,
            num_heads,
            head_dim,
            base_theta
        );
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        nvidia::rope(
            output->data(),
            input->data(),
            reinterpret_cast<const int64_t*>(position_ids->data()), // ✅ 修改处：显式转换类型
            output->dtype(),
            seq_len,
            num_heads,
            head_dim,
            base_theta
        );
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace llaisys::ops