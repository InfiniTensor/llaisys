#include "rope_nvidia.hpp"

#include "../../nvidia_cuda.cuh"

#include <cmath>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void rope_kernel(
    T *out,
    const T *in,
    const int64_t *pos_ids,
    size_t seqlen,
    size_t nhead,
    size_t d,
    float theta) {
    size_t half_d = d / 2;
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = seqlen * nhead * half_d;
    if (idx >= total) {
        return;
    }

    size_t j = idx % half_d;
    size_t tmp = idx / half_d;
    size_t h = tmp % nhead;
    size_t i = tmp / nhead;

    double pos = static_cast<double>(pos_ids[i]);
    double exponent = (2.0 * static_cast<double>(j)) / static_cast<double>(d);
    double angle = pos / pow(static_cast<double>(theta), exponent);
    double c = cos(angle);
    double s = sin(angle);

    size_t base = (i * nhead + h) * d;
    size_t idx_a = base + j;
    size_t idx_b = base + j + half_d;

    double a = static_cast<double>(to_float(in[idx_a]));
    double b = static_cast<double>(to_float(in[idx_b]));

    out[idx_a] = from_float<T>(static_cast<float>(a * c - b * s));
    out[idx_b] = from_float<T>(static_cast<float>(b * c + a * s));
}

template <typename T>
void launch_rope(
    std::byte *out,
    const std::byte *in,
    const std::byte *pos_ids,
    size_t seqlen,
    size_t nhead,
    size_t d,
    float theta) {
    size_t total = seqlen * nhead * (d / 2);
    const int threads = num_threads_1d();
    const int blocks = num_blocks_1d(total, threads);
    auto stream = current_stream();
    rope_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(in),
        reinterpret_cast<const int64_t *>(pos_ids),
        seqlen,
        nhead,
        d,
        theta);
    check_cuda(cudaGetLastError(), "rope_kernel launch");
}

void rope(
    std::byte *out,
    const std::byte *in,
    const std::byte *pos_ids,
    llaisysDataType_t type,
    size_t seqlen,
    size_t nhead,
    size_t d,
    float theta) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launch_rope<float>(out, in, pos_ids, seqlen, nhead, d, theta);
    case LLAISYS_DTYPE_F16:
        return launch_rope<llaisys::fp16_t>(out, in, pos_ids, seqlen, nhead, d, theta);
    case LLAISYS_DTYPE_BF16:
        return launch_rope<llaisys::bf16_t>(out, in, pos_ids, seqlen, nhead, d, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia
