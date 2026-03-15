#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/rope_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/rope_metax.hpp"
#endif

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    // 1. 参数校验
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
           "RoPE: all tensors must be contiguous.");

    CHECK_ARGUMENT(out->ndim() == 3, "RoPE: out must be 3D [seqlen, nhead, d].");
    CHECK_ARGUMENT(pos_ids->ndim() == 1, "RoPE: pos_ids must be 1D [seqlen].");
    CHECK_ARGUMENT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "RoPE: pos_ids must be int64.");
    CHECK_ARGUMENT(theta > 0.0f, "RoPE: theta must be positive.");

    const size_t seqlen = out->shape()[0];
    const size_t nhead = out->shape()[1];
    const size_t d = out->shape()[2];
    CHECK_ARGUMENT((d % 2) == 0, "RoPE: head_dim must be even.");
    CHECK_ARGUMENT(pos_ids->shape()[0] == seqlen, "RoPE: pos_ids shape must match seqlen.");

    // 2. 设置设备上下文
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(),
                         in->data(),
                         reinterpret_cast<const int64_t *>(pos_ids->data()),
                         out->dtype(),
                         seqlen,
                         nhead,
                         d,
                         theta);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rope(out->data(),
                            in->data(),
                            reinterpret_cast<const int64_t *>(pos_ids->data()),
                            out->dtype(),
                            seqlen,
                            nhead,
                            d,
                            theta);
#endif
#ifdef ENABLE_METAX_API
    case LLAISYS_DEVICE_METAX:
        return metax::rope(out->data(),
                           in->data(),
                           reinterpret_cast<const int64_t *>(pos_ids->data()),
                           out->dtype(),
                           seqlen,
                           nhead,
                           d,
                           theta);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
