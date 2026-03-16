#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cmath>
#ifdef ENABLE_NVIDIA_API
#include "llaisys/ops_nvidia.h"
#endif

namespace {

// out_i = up_i * gate_i / (1 + exp(-gate_i)) = up_i * silu(gate_i)，逐元素；分段计算避免 exp 溢出
inline float silu(float x) {
    if (x >= 0.f) {
        return x / (1.f + std::exp(-x));
    }
    float e = std::exp(x);
    return x * e / (1.f + e);
}

template <typename T>
void swiglu_impl(T *out, const T *gate, const T *up, size_t n) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < n; i++) {
        float g, u;
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            g = llaisys::utils::cast<float>(gate[i]);
            u = llaisys::utils::cast<float>(up[i]);
        } else {
            g = static_cast<float>(gate[i]);
            u = static_cast<float>(up[i]);
        }
        float y = u * silu(g);
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>)
            out[i] = llaisys::utils::cast<T>(y);
        else
            out[i] = static_cast<T>(y);
    }
}

void swiglu_cpu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t dtype,
                size_t n) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        swiglu_impl(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(gate),
                    reinterpret_cast<const float *>(up), n);
        return;
    case LLAISYS_DTYPE_F16:
        swiglu_impl(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(gate),
                    reinterpret_cast<const llaisys::fp16_t *>(up), n);
        return;
    case LLAISYS_DTYPE_BF16:
        swiglu_impl(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(gate),
                    reinterpret_cast<const llaisys::bf16_t *>(up), n);
        return;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);
    ASSERT(out->dtype() == gate->dtype() && gate->dtype() == up->dtype(),
           "swiglu: out, gate, up must have same dtype");
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(),
           "swiglu: out, gate, up must be contiguous");
    ASSERT(out->ndim() == 2 && gate->ndim() == 2 && up->ndim() == 2, "swiglu: all must be 2D");
    ASSERT(out->shape() == gate->shape() && gate->shape() == up->shape(),
           "swiglu: out, gate, up must have same shape");
    size_t n = out->numel();

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return swiglu_cpu(out->data(), gate->data(), up->data(), out->dtype(), n);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        nvidia::swiglu(out->data(), gate->data(), up->data(), out->dtype(), n);
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
