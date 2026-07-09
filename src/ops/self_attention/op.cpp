#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/flash_attention_cpu.hpp"
#include "cpu/self_attention_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/flash_attention_cuda.cuh"
#include "nvidia/self_attention_cuda.cuh"
#endif

// 定义此宏以启用 Flash Attention 2 后端
// 对序列较长时内存占用和速度均有提升
#define USE_FLASH_ATTENTION

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    // 检查设备一致性
    CHECK_SAME_DEVICE(attn_val, q, k, v);

    // 检查数据类型一致性
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());

    // 检查维度
    CHECK_ARGUMENT(q->ndim() == 3, "self_attention: q must be 3D tensor [seqlen, nhead, d]");
    CHECK_ARGUMENT(k->ndim() == 3, "self_attention: k must be 3D tensor [total_len, nkvhead, d]");
    CHECK_ARGUMENT(v->ndim() == 3, "self_attention: v must be 3D tensor [total_len, nkvhead, dv]");
    CHECK_ARGUMENT(attn_val->ndim() == 3, "self_attention: attn_val must be 3D tensor [seqlen, nhead, dv]");

    // 获取形状参数
    size_t seq_len = q->shape()[0];
    size_t n_heads = q->shape()[1];
    size_t head_dim = q->shape()[2];

    size_t total_len = k->shape()[0];
    size_t n_kv_heads = k->shape()[1];
    size_t k_head_dim = k->shape()[2];

    size_t v_total_len = v->shape()[0];
    size_t v_kv_heads = v->shape()[1];
    size_t v_head_dim = v->shape()[2];

    // 检查形状兼容性
    CHECK_ARGUMENT(k_head_dim == head_dim, "self_attention: k and q must have same head dimension");
    CHECK_ARGUMENT(total_len == v_total_len, "self_attention: k and v must have same sequence length");
    CHECK_ARGUMENT(n_kv_heads == v_kv_heads, "self_attention: k and v must have same number of kv heads");
    CHECK_ARGUMENT(n_heads % n_kv_heads == 0, "self_attention: n_heads must be divisible by n_kv_heads");

    CHECK_ARGUMENT(attn_val->shape()[0] == seq_len, "self_attention: attn_val seq_len must match q");
    CHECK_ARGUMENT(attn_val->shape()[1] == n_heads, "self_attention: attn_val n_heads must match q");
    CHECK_ARGUMENT(attn_val->shape()[2] == v_head_dim, "self_attention: attn_val head_dim must match v");

    // 检查所有张量都是连续的
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "self_attention: all tensors must be contiguous.");

    // 设置设备上下文
    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
#ifdef USE_FLASH_ATTENTION
        return cpu::flash_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                    attn_val->dtype(), seq_len, total_len,
                                    n_heads, n_kv_heads, v_head_dim, scale);
#else
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                   attn_val->dtype(), seq_len, total_len, n_heads, n_kv_heads, v_head_dim, scale);
#endif
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
#ifdef USE_FLASH_ATTENTION
        return nvidia::flash_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                       attn_val->dtype(), seq_len, total_len,
                                       n_heads, n_kv_heads, v_head_dim, scale);
#else
        return nvidia::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                      attn_val->dtype(), seq_len, total_len,
                                      n_heads, n_kv_heads, v_head_dim, scale);
#endif
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
