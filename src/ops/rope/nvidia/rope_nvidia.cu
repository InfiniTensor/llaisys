#include "rope_nvidia.hpp"

#include "../../../device/nvidia/cuda_utils.hpp"

#include <cmath>

namespace llaisys::ops::nvidia {
namespace {
template <typename T>
__global__ void rope_kernel(T *out, const T *in, const int64_t *pos_ids, size_t seqlen, size_t nhead, size_t dim,
                            float theta) {
    size_t half = dim / 2;
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = seqlen * nhead * half;
    if (idx >= total) {
        return;
    }
    size_t j = idx % half;
    size_t tmp = idx / half;
    size_t h = tmp % nhead;
    size_t s = tmp / nhead;
    float p = static_cast<float>(pos_ids[s]);
    float exponent = 2.0f * static_cast<float>(j) / static_cast<float>(dim);
    float angle = p / powf(theta, exponent);
    float sinv = sinf(angle);
    float cosv = cosf(angle);

    size_t base = (s * nhead + h) * dim;
    float a = llaisys::device::nvidia::ScalarOps<T>::load(in + base + j);
    float b = llaisys::device::nvidia::ScalarOps<T>::load(in + base + half + j);
    llaisys::device::nvidia::ScalarOps<T>::store(out + base + j, a * cosv - b * sinv);
    llaisys::device::nvidia::ScalarOps<T>::store(out + base + half + j, b * cosv + a * sinv);
}

template <typename T>
void launch_rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, size_t seqlen, size_t nhead,
                 size_t dim, float theta) {
    size_t half = dim / 2;
    size_t total = seqlen * nhead * half;
    const int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    rope_kernel<<<blocks, threads>>>(reinterpret_cast<T *>(out), reinterpret_cast<const T *>(in),
                                     reinterpret_cast<const int64_t *>(pos_ids), seqlen, nhead, dim, theta);
    llaisys::device::nvidia::cuda_check(cudaGetLastError());
}
} // namespace

void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, llaisysDataType_t type, size_t seqlen,
          size_t nhead, size_t dim, float theta) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launch_rope<float>(out, in, pos_ids, seqlen, nhead, dim, theta);
    case LLAISYS_DTYPE_BF16:
        return launch_rope<llaisys::bf16_t>(out, in, pos_ids, seqlen, nhead, dim, theta);
    case LLAISYS_DTYPE_F16:
        return launch_rope<llaisys::fp16_t>(out, in, pos_ids, seqlen, nhead, dim, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
