#include "rope_nvidia.cuh"

#include "../../nvidia/nvidia_common.cuh"

#include <cmath>

namespace {

template <typename T>
__global__ void rope_kernel(T *out,
                            const T *in,
                            const int64_t *pos_ids,
                            double log_theta,
                            int64_t num_heads,
                            int64_t head_dim,
                            size_t total_pairs) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
    if (idx >= total_pairs) {
        return;
    }

    const int64_t half_dim = head_dim / 2;
    const int64_t pairs_per_seq = num_heads * half_dim;
    const int64_t seq = static_cast<int64_t>(idx / static_cast<size_t>(pairs_per_seq));
    const int64_t rem = static_cast<int64_t>(idx % static_cast<size_t>(pairs_per_seq));
    const int64_t head = rem / half_dim;
    const int64_t dim = rem % half_dim;
    const int64_t base = (seq * num_heads + head) * head_dim;

    const float a = llaisys::device::nvidia::cuda_utils::toFloat<T>(in[base + dim]);
    const float b = llaisys::device::nvidia::cuda_utils::toFloat<T>(in[base + half_dim + dim]);
    const double exponent = -2.0 * static_cast<double>(dim) / static_cast<double>(head_dim);
    const double freq = exp(log_theta * exponent);
    const double angle = static_cast<double>(pos_ids[seq]) * freq;
    const float sin_v = static_cast<float>(sin(angle));
    const float cos_v = static_cast<float>(cos(angle));

    out[base + dim] = llaisys::device::nvidia::cuda_utils::fromFloat<T>(a * cos_v - b * sin_v);
    out[base + half_dim + dim] = llaisys::device::nvidia::cuda_utils::fromFloat<T>(b * cos_v + a * sin_v);
}

template <typename T>
void rope_impl(std::byte *out,
               const std::byte *in,
               const int64_t *pos_ids,
               float theta,
               const std::vector<size_t> &shape) {
    const int64_t seq_len = static_cast<int64_t>(shape[0]);
    const int64_t num_heads = static_cast<int64_t>(shape[1]);
    const int64_t head_dim = static_cast<int64_t>(shape[2]);
    const size_t total_pairs = static_cast<size_t>(seq_len * num_heads * (head_dim / 2));
    constexpr int threads = 256;
    rope_kernel<<<llaisys::ops::nvidia::blocks_for(total_pairs, threads), threads, 0, llaisys::ops::nvidia::current_stream()>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(in),
        pos_ids,
        log(static_cast<double>(theta)),
        num_heads,
        head_dim,
        total_pairs);
    LLAISYS_CUDA_CHECK(cudaGetLastError());
}

} // namespace

namespace llaisys::ops::nvidia {

void rope(std::byte *out,
          const std::byte *in,
          const int64_t *pos_ids,
          float theta,
          llaisysDataType_t dtype,
          const std::vector<size_t> &shape) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rope_impl<float>(out, in, pos_ids, theta, shape);
    case LLAISYS_DTYPE_F16:
        return rope_impl<llaisys::fp16_t>(out, in, pos_ids, theta, shape);
    case LLAISYS_DTYPE_BF16:
        return rope_impl<llaisys::bf16_t>(out, in, pos_ids, theta, shape);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia
