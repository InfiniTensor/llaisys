#include "rope_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>
#include <cstddef>
namespace llaisys::ops::cpu {
template<typename T>
void rope_t(std::byte* out_raw,
            const std::byte* in_raw,
            const int64_t* pos_raw,
            size_t seqlen,
            size_t nhead,
            size_t head_dim,
            float theta) {
    const T* x = reinterpret_cast<const T*>(in_raw);
    T* y = reinterpret_cast<T*>(out_raw);
    size_t half = head_dim / 2;

    for (size_t i = 0; i < seqlen; ++i) {
        float p = static_cast<float>(pos_raw[i]);
        for (size_t h = 0; h < nhead; ++h) {
            const T* x_vec = x + (i * nhead + h) * head_dim;
            T* y_vec = y + (i * nhead + h) * head_dim;

            for (size_t j = 0; j < half; ++j) {
                // phi_{i,j} = p / theta^(2j / d)
                float exp_term = static_cast<float>(2.0f * j) / static_cast<float>(head_dim);
                float angle = p / std::pow(theta, exp_term);
                float s = std::sin(angle);
                float c = std::cos(angle);

                float a = llaisys::utils::cast<float>(x_vec[j]);
                float b = llaisys::utils::cast<float>(x_vec[half + j]);

                y_vec[j] = llaisys::utils::cast<T>(a * c - b * s);           // a'
                y_vec[half + j] = llaisys::utils::cast<T>(b * c + a * s);    // b'
            }
        }
    }
}

void rope(std::byte* out,
            const std::byte* in,
            const std::byte* pos_ids,
            llaisysDataType_t dtype,
            size_t seqlen,
            size_t nhead,
            size_t head_dim,
            float theta) {
    auto* pos_ptr = reinterpret_cast<const int64_t*>(pos_ids);
    switch (dtype)
    {
    case LLAISYS_DTYPE_F32:
        return rope_t<float>(out, in, pos_ptr, seqlen, nhead, head_dim, theta);
    case LLAISYS_DTYPE_F16:
        return rope_t<llaisys::fp16_t>(out, in, pos_ptr, seqlen, nhead, head_dim, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_t<llaisys::bf16_t>(out, in, pos_ptr, seqlen, nhead, head_dim, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);

    }

}
} // namespace llaisys::ops::cpu