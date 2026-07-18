#include "rope_nvidia.cuh"

#include "../../../utils.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <type_traits>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void rope_kernel(T *out, const T *in, const int64_t *pos_ids, float theta,
                            size_t seq_len, size_t n_heads, size_t head_dim, size_t half_dim) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t numel = seq_len * n_heads * half_dim;
    if (idx >= numel) {
        return;
    }

    size_t k = idx % half_dim;
    size_t tmp = idx / half_dim;
    size_t head = tmp % n_heads;
    size_t seq = tmp / n_heads;

    size_t offset = seq * n_heads * head_dim + head * head_dim;
    float angle = static_cast<float>(pos_ids[seq]) / powf(theta, 2.0f * static_cast<float>(k) / static_cast<float>(head_dim));
    float sin_angle = sinf(angle);
    float cos_angle = cosf(angle);

    if constexpr (std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t>) {
        float a = utils::cast_device<float>(in[offset + k]);
        float b = utils::cast_device<float>(in[offset + k + half_dim]);
        out[offset + k] = utils::cast_device<T>(a * cos_angle - b * sin_angle);
        out[offset + k + half_dim] = utils::cast_device<T>(b * cos_angle + a * sin_angle);
    } else {
        T a = in[offset + k];
        T b = in[offset + k + half_dim];
        out[offset + k] = a * cos_angle - b * sin_angle;
        out[offset + k + half_dim] = b * cos_angle + a * sin_angle;
    }
}

void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          float theta, llaisysDataType_t dtype,
          size_t seq_len, size_t n_heads, size_t head_dim) {
    const size_t half_dim = head_dim / 2;
    const size_t numel = seq_len * n_heads * half_dim;
    const int block_size = 256;
    const int num_blocks = static_cast<int>((numel + block_size - 1) / block_size);

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        rope_kernel<<<num_blocks, block_size>>>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const int64_t *>(pos_ids),
            theta, seq_len, n_heads, head_dim, half_dim);
        break;
    case LLAISYS_DTYPE_F16:
        rope_kernel<<<num_blocks, block_size>>>(
            reinterpret_cast<fp16_t *>(out),
            reinterpret_cast<const fp16_t *>(in),
            reinterpret_cast<const int64_t *>(pos_ids),
            theta, seq_len, n_heads, head_dim, half_dim);
        break;
    case LLAISYS_DTYPE_BF16:
        rope_kernel<<<num_blocks, block_size>>>(
            reinterpret_cast<bf16_t *>(out),
            reinterpret_cast<const bf16_t *>(in),
            reinterpret_cast<const int64_t *>(pos_ids),
            theta, seq_len, n_heads, head_dim, half_dim);
        break;
    default:
        std::fprintf(stderr, "[ERROR] Unsupported data type for CUDA rope: %d\n", dtype);
        throw std::runtime_error("Unsupported data type");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[CUDA ERROR] rope kernel launch failed: %s\n", cudaGetErrorString(err));
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

} // namespace llaisys::ops::nvidia
