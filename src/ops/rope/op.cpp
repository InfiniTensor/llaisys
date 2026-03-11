#include "op.hpp"

#include "cpu/rope_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "../nvidia/ops_nvidia.cuh"
#endif

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    if (pos_ids->dtype() != LLAISYS_DTYPE_I64) {
        throw std::runtime_error("RoPE: pos_ids must be Int64");
    }

    size_t seq_len = in->shape()[0];
    size_t n_head = in->shape()[1];
    size_t head_dim = in->shape()[2];
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(), theta, seq_len, n_head, head_dim, out->dtype());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rope(out->data(), in->data(), pos_ids->data(), theta, seq_len, n_head, head_dim, out->dtype());
#endif
    default:
        throw std::runtime_error("RoPE: device not supported");
    }
}
} // namespace llaisys::ops
