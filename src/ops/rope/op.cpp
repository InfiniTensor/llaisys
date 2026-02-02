#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_ARGUMENT(out->ndim() == 3, "out must be a 3D tensor");
    CHECK_ARGUMENT(in->ndim() == 3, "in must be a 3D tensor");
    CHECK_ARGUMENT(pos_ids->ndim() == 1, "pos_ids must be a 1D tensor");
    CHECK_ARGUMENT(out->shape()[0] == in->shape()[0], "out and in must have the same sequence length");
    CHECK_ARGUMENT(out->shape()[1] == in->shape()[1], "out and in must have the same number of heads");
    CHECK_ARGUMENT(out->shape()[2] == in->shape()[2], "out and in must have the same head dimension");
    CHECK_ARGUMENT(pos_ids->shape()[0] == in->shape()[0], "pos_ids length must match sequence length");
    CHECK_ARGUMENT(out->isContiguous(), "rope: out tensor must be contiguous.");
    CHECK_ARGUMENT(in->isContiguous(), "rope: in tensor must be contiguous.");
    CHECK_ARGUMENT(pos_ids->isContiguous(), "rope: pos_ids tensor must be contiguous.");

    size_t seq_len = in->shape()[0];
    size_t num_heads = in->shape()[1];
    size_t head_dim = in->shape()[2];

    // 总是支持CPU计算
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(
            out->data(), 
            in->data(), 
            pos_ids->data(), 
            theta, 
            out->dtype(), 
            seq_len, 
            num_heads, 
            head_dim
        );
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(
            out->data(), 
            in->data(), 
            pos_ids->data(), 
            theta, 
            out->dtype(), 
            seq_len, 
            num_heads, 
            head_dim
        );
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
