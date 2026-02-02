#include "rms_norm_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>
#include <cstddef>

namespace llaisys::ops::cpu {

template <typename T>
void rms_norm_t(std::byte* out_raw,
                 const std::byte* in_raw,
                 const std::byte* w_raw,
                 size_t batch,
                 size_t dim,
                 float eps) {
    const T* x = reinterpret_cast<const T*>(in_raw);
    const T* w = reinterpret_cast<const T*>(w_raw);
    T* y = reinterpret_cast<T*>(out_raw);

    for (size_t i = 0; i < batch; ++i)
    {
        const T* x_row = x + i * dim;
        T* y_row = y + i * dim;

        float sum_sq = 0.0f;

        for (size_t j = 0; j < dim; ++j)
        {
            float val = llaisys::utils::cast<float>(x_row[j]);
            sum_sq += val * val;
        }

        float  inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(dim) + eps);

        for (size_t j = 0; j < dim; ++j)
        {
            float val = llaisys::utils::cast<float>(x_row[j]) * inv_rms;
            float w_val = llaisys::utils::cast<float>(w[j]);
            y_row[j] = llaisys::utils::cast<T>(val * w_val);
        }
    }
}

void rms_norm(std::byte* out,
                  const std::byte* in,
                  const std::byte* weight,
                  llaisysDataType_t dtype,
                  size_t batch,
                  size_t dim,
                  float eps) {
    switch (dtype)
    {
    case LLAISYS_DTYPE_F32:
        rms_norm_t<float>(out, in, weight, batch, dim, eps);
        break;
    case LLAISYS_DTYPE_F16:
        rms_norm_t<llaisys::fp16_t>(out, in, weight, batch, dim, eps);
        break;
        case LLAISYS_DTYPE_BF16:
        rms_norm_t<llaisys::bf16_t>(out, in, weight, batch, dim, eps);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} //namespace llaisys::ops::cpu
