#include "linear_cpu.hpp"
#include "../../../utils.hpp"

namespace {

template <typename T>
inline void linear_impl(T *y,
                        const T *x,
                        const T *w,
                        const T *b, // may be nullptr
                        size_t N, size_t M, size_t K) {
    // Row-major contiguous:
    // x: [N,K], w: [M,K], y: [N,M]
    for (size_t i = 0; i < N; ++i) {
        const T *x_row = x + i * K;
        T *y_row = y + i * M;
        for (size_t j = 0; j < M; ++j) {
            const T *w_row = w + j * K; // W[j, :]
            float acc = 0.f;
            if (b) {
                acc += llaisys::utils::cast<float>(b[j]);
            }
            for (size_t k = 0; k < K; ++k) {
                float xv = llaisys::utils::cast<float>(x_row[k]);
                float wv = llaisys::utils::cast<float>(w_row[k]);
                acc += xv * wv;
            }
            y_row[j] = llaisys::utils::cast<T>(acc);
        }
    }
}

} // anonymous namespace

namespace llaisys::ops::cpu {

void linear(std::byte *out,
            const std::byte *in,
            const std::byte *weight,
            const std::byte *bias,
            llaisysDataType_t type,
            size_t N, size_t M, size_t K) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_impl(reinterpret_cast<float *>(out),
                           reinterpret_cast<const float *>(in),
                           reinterpret_cast<const float *>(weight),
                           reinterpret_cast<const float *>(bias),
                           N, M, K);
    case LLAISYS_DTYPE_F16:
        return linear_impl(reinterpret_cast<llaisys::fp16_t *>(out),
                           reinterpret_cast<const llaisys::fp16_t *>(in),
                           reinterpret_cast<const llaisys::fp16_t *>(weight),
                           reinterpret_cast<const llaisys::fp16_t *>(bias),
                           N, M, K);
    case LLAISYS_DTYPE_BF16:
        return linear_impl(reinterpret_cast<llaisys::bf16_t *>(out),
                           reinterpret_cast<const llaisys::bf16_t *>(in),
                           reinterpret_cast<const llaisys::bf16_t *>(weight),
                           reinterpret_cast<const llaisys::bf16_t *>(bias),
                           N, M, K);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu
