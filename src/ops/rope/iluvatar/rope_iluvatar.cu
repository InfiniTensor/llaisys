#include "rope_iluvatar.cuh"

#include "../../../device/iluvatar/cuda_utils.cuh"
#include "../../../utils.hpp"

#include <cuda_runtime.h>
#include <cmath>

namespace llaisys::ops::iluvatar {

template <typename T>
__global__ void ropeKernel(T *out, const T *in, const int64_t *pos_ids, float theta,
                           size_t seq_len, size_t nhead, size_t d) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = seq_len * nhead * d;
    
    if (idx >= total) return;
    
    // Compute indices
    size_t tmp = idx;
    size_t d_idx = tmp % d;
    tmp /= d;
    size_t h = tmp % nhead;
    size_t s = tmp / nhead;
    
    // Get position id
    int64_t pos = pos_ids[s];
    
    // RoPE computation
    size_t half_d = d / 2;
    size_t j = d_idx % half_d;
    
    // Compute angle
    float angle = pos / powf(theta, (2.0f * j) / d);
    
    // Load input values
    float x_d = to_float_cuda(in[idx]);
    
    // Determine if we're in the first or second half
    float result;
    if (d_idx < half_d) {
        // a'_i,j = a_i,j * cos(angle) - b_i,j * sin(angle)
        float x_d_plus_half = to_float_cuda(in[s * nhead * d + h * d + d_idx + half_d]);
        result = x_d * cosf(angle) - x_d_plus_half * sinf(angle);
    } else {
        // b'_i,j = b_i,j * cos(angle) + a_i,j * sin(angle)
        float x_d_minus_half = to_float_cuda(in[s * nhead * d + h * d + d_idx - half_d]);
        result = x_d * cosf(angle) + x_d_minus_half * sinf(angle);
    }
    
    out[idx] = from_float_cuda<T>(result);
}

template <typename T>
void rope_(std::byte *out_bytes, const std::byte *in_bytes, const std::byte *pos_ids_bytes, 
           float theta, size_t seq_len, size_t nhead, size_t d) {
    auto out = reinterpret_cast<T *>(out_bytes);
    auto in = reinterpret_cast<const T *>(in_bytes);
    auto pos_ids = reinterpret_cast<const int64_t *>(pos_ids_bytes);
    
    size_t total = seq_len * nhead * d;
    const int blockSize = 256;
    const int numBlocks = (total + blockSize - 1) / blockSize;
    
    ropeKernel<T><<<numBlocks, blockSize>>>(out, in, pos_ids, theta, seq_len, nhead, d);
}

void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, float theta,
          llaisysDataType_t type, size_t seq_len, size_t nhead, size_t d) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_<float>(out, in, pos_ids, theta, seq_len, nhead, d);
    case LLAISYS_DTYPE_BF16:
        return rope_<bf16_t_cuda>(out, in, pos_ids, theta, seq_len, nhead, d);
    case LLAISYS_DTYPE_F16:
        return rope_<fp16_t_cuda>(out, in, pos_ids, theta, seq_len, nhead, d);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

}
