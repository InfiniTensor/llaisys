#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/rope_cuda.cuh"
#endif

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    // 检查设备一致性
    CHECK_SAME_DEVICE(out, in, pos_ids);

    // 检查数据类型
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_ARGUMENT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "rope: pos_ids must be int64 type");

    // 检查维度
    CHECK_ARGUMENT(in->ndim() == 3, "rope: input must be 3D tensor [seqlen, nhead, d]");
    CHECK_ARGUMENT(out->ndim() == 3, "rope: output must be 3D tensor [seqlen, nhead, d]");
    CHECK_ARGUMENT(pos_ids->ndim() == 1, "rope: pos_ids must be 1D tensor");

    // 检查形状兼容性
    size_t seq_len = in->shape()[0];
    size_t n_heads = in->shape()[1];
    size_t head_dim = in->shape()[2];

    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_ARGUMENT(pos_ids->shape()[0] == seq_len,
                   "rope: pos_ids length must match sequence length");
    CHECK_ARGUMENT(head_dim % 2 == 0, "rope: head dimension must be even");

    // 检查所有张量都是连续的
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
           "rope: all tensors must be contiguous.");

    // 设置设备上下文
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(),
                         out->dtype(), seq_len, n_heads, head_dim, theta);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rope(out->data(), in->data(), pos_ids->data(),
                            out->dtype(), seq_len, n_heads, head_dim, theta);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
