#include "op.hpp"

#include "cpu/rms_norm_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "../nvidia/ops_nvidia.cuh"
#endif

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    size_t hidden_size = in->shape().back();
    size_t num_rows = in->numel() / hidden_size;

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), weight->data(), eps, num_rows, hidden_size, out->dtype());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rms_norm(out->data(), in->data(), weight->data(), eps, num_rows, hidden_size, out->dtype());
#endif
    default:
        throw std::runtime_error("RMSNorm: device not supported");
    }
}
} // namespace llaisys::ops
