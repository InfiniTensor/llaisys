#include "rope_nvidia.cuh"

#include "../../../device/nvidia/cuda_utils.cuh"
#include "../../../utils.hpp"

#include <cuda_runtime.h>
#include <cmath>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void ropeKernel(T *out, const T *in, const int64_t *pos_ids, float theta,
                           size_t seq_len, size_t num_heads, size_t head_dim) {
    size_t half_dim = head_dim / 2;

    // Each thread processes one (seq, head, half_dim) position
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_pos = seq_len * num_heads * half_dim;

    if (idx >= total_pos) return;

    size_t h = idx % half_dim;
    size_t tmp = idx / half_dim;
    size_t head = tmp % num_heads;
    size_t s = tmp / num_heads;

    int64_t p = pos_ids[s];

    // Compute rotation angle
    float dim = static_cast<float>(h);
    float exponent = 2.0f * dim / head_dim;
    float theta_pow = powf(theta, exponent);
    float phi = static_cast<float>(p) / theta_pow;

    float cos_phi = cosf(phi);
    float sin_phi = sinf(phi);

    // Get input values [a, b]
    size_t index_a = s * num_heads * head_dim + head * head_dim + h;
    size_t index_b = index_a + half_dim;

    float a = to_float_cuda(in[index_a]);
    float b = to_float_cuda(in[index_b]);

    // Apply rotation
    float a_prime = a * cos_phi - b * sin_phi;
    float b_prime = b * cos_phi + a * sin_phi;

    // Store result
    out[index_a] = from_float_cuda<T>(a_prime);
    out[index_b] = from_float_cuda<T>(b_prime);
}

template <typename T>
void rope_(std::byte *out_bytes, const std::byte *in_bytes, const std::byte *pos_ids_bytes,
           float theta, size_t seq_len, size_t num_heads, size_t head_dim) {
    auto out = reinterpret_cast<T *>(out_bytes);
    auto in = reinterpret_cast<const T *>(in_bytes);
    auto pos_ids = reinterpret_cast<const int64_t *>(pos_ids_bytes);

    size_t half_dim = head_dim / 2;
    size_t total_pos = seq_len * num_heads * half_dim;

    const int blockSize = 256;
    const int numBlocks = (total_pos + blockSize - 1) / blockSize;

    ropeKernel<T><<<numBlocks, blockSize>>>(out, in, pos_ids, theta, seq_len, num_heads, head_dim);
}

void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          llaisysDataType_t type, float theta, size_t seq_len, size_t num_heads, size_t head_dim) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_<float>(out, in, pos_ids, theta, seq_len, num_heads, head_dim);
    case LLAISYS_DTYPE_BF16:
        return rope_<bf16_t_cuda>(out, in, pos_ids, theta, seq_len, num_heads, head_dim);
    case LLAISYS_DTYPE_F16:
        return rope_<fp16_t_cuda>(out, in, pos_ids, theta, seq_len, num_heads, head_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

}
