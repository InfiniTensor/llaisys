#include "op.hpp"

#include "cpu/swiglu_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "../nvidia/ops_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "../metax/ops_metax.cuh"
#endif

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    size_t numel = out->numel();
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::swiglu(out->data(), gate->data(), up->data(), numel, out->dtype());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::swiglu(out->data(), gate->data(), up->data(), numel, out->dtype());
#endif
#ifdef ENABLE_METAX_API
    case LLAISYS_DEVICE_METAX:
        return metax::swiglu(out->data(), gate->data(), up->data(), numel, out->dtype());
#endif
    default:
        throw std::runtime_error("SwiGLU: device not supported");
    }
}
} // namespace llaisys::ops
