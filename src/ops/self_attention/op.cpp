#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/self_attention_nvidia.hpp"
#endif

namespace llaisys::ops {

// 执行自注意力计算（支持标准 MHA 和 GQA）
// 输入张量布局：[seq_len, num_heads, head_dim]
// 要求 Q/K/V 和输出位于同一设备且数据类型一致
void self_attention(tensor_t output, tensor_t query, tensor_t key, tensor_t value, float softmax_scale) {
    // 检查所有张量是否在同一设备
    CHECK_SAME_DEVICE(output, query, key, value);
    // 数据类型必须完全一致
    CHECK_SAME_DTYPE(output->dtype(), query->dtype(), key->dtype(), value->dtype());

    // 所有输入和输出必须是三维张量
    ASSERT(output->ndim() == 3 && query->ndim() == 3 && key->ndim() == 3 && value->ndim() == 3,
           "SelfAttention: all tensors must be 3-dimensional with shape [seq_len, num_heads, head_dim].");

    size_t query_len = query->shape()[0];
    size_t num_q_heads = query->shape()[1];
    size_t qk_head_dim = query->shape()[2];

    size_t kv_len = key->shape()[0];
    size_t num_kv_heads = key->shape()[1];
    size_t k_head_dim = key->shape()[2];
    size_t v_head_dim = value->shape()[2];

    // 验证 Q 与 K 的头维度一致
    ASSERT(qk_head_dim == k_head_dim,
           "SelfAttention: query and key must have the same head dimension.");
    // 验证 V 与 K 的序列长度和 KV 头数一致
    ASSERT(value->shape()[0] == kv_len && value->shape()[1] == num_kv_heads,
           "SelfAttention: value tensor shape must match key in sequence length and number of KV heads.");
    // 验证输出形状匹配预期：[query_len, num_q_heads, v_head_dim]
    ASSERT(output->shape()[0] == query_len &&
           output->shape()[1] == num_q_heads &&
           output->shape()[2] == v_head_dim,
           "SelfAttention: output tensor shape does not match expected dimensions.");
    // 支持 GQA：Q 头数必须是 KV 头数的整数倍
    ASSERT(num_q_heads % num_kv_heads == 0,
           "SelfAttention: number of query heads must be divisible by number of KV heads (for GQA support).");

    // 所有张量必须内存连续
    ASSERT(output->isContiguous() && query->isContiguous() && key->isContiguous() && value->isContiguous(),
           "SelfAttention: all tensors must be contiguous in memory.");

    // CPU 路径直接返回
    if (output->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(
            output->data(),
            query->data(),
            key->data(),
            value->data(),
            output->dtype(),
            query_len,
            kv_len,
            num_q_heads,
            num_kv_heads,
            qk_head_dim,
            v_head_dim,
            softmax_scale
        );
    }

    // 设置当前 CUDA 设备
    llaisys::core::context().setDevice(output->deviceType(), output->deviceId());

    switch (output->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        // 理论上不会执行到这里，因为上面已经处理了 CPU 情况
        return cpu::self_attention(
            output->data(),
            query->data(),
            key->data(),
            value->data(),
            output->dtype(),
            query_len,
            kv_len,
            num_q_heads,
            num_kv_heads,
            qk_head_dim,
            v_head_dim,
            softmax_scale
        );
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        nvidia::self_attention(
            output->data(),
            query->data(),
            key->data(),
            value->data(),
            output->dtype(),
            query_len,
            kv_len,
            num_q_heads,
            num_kv_heads,
            qk_head_dim,
            v_head_dim,
            softmax_scale
        );
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace llaisys::ops