#include "rope_metax.cuh"

#include "../../../device/metax/cuda_utils.cuh"
#include "../../../utils.hpp"

#include <cuda_runtime.h>
#include <cmath>

namespace llaisys::ops::metax {

template <typename T>
__global__ void ropeKernel(T *out, const T *in, const int64_t *pos_ids, float theta,
                           size_t seqlen, size_t nhead, size_t d) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = seqlen * nhead * d;

    if (idx >= total) return;

    // Calculate indices
    size_t i = idx / (nhead * d);       // sequence index
    size_t h = (idx / d) % nhead;       // head index
    size_t j = idx % d;                 // dimension index

    int64_t pos = pos_ids[i];

    // Split dimension into two halves
    size_t half_d = d / 2;

    // Get the position in the first or second half
    if (j < half_d) {
        // First half: a (cos part)
        float angle = pos / powf(theta, (2.0f * j) / d);
        float cos_val = cosf(angle);
        float sin_val = sinf(angle);

        float a = to_float_metax(in[i * nhead * d + h * d + j]);
        float b = to_float_metax(in[i * nhead * d + h * d + j + half_d]);

        float result = a * cos_val - b * sin_val;
        out[idx] = from_float_metax<T>(result);
    } else {
        // Second half: b (sin part)
        size_t j_in_half = j - half_d;
        float angle = pos / powf(theta, (2.0f * j_in_half) / d);
        float cos_val = cosf(angle);
        float sin_val = sinf(angle);

        float a = to_float_metax(in[i * nhead * d + h * d + j_in_half]);
        float b = to_float_metax(in[i * nhead * d + h * d + j]);

        float result = b * cos_val + a * sin_val;
        out[idx] = from_float_metax<T>(result);
    }
}

template <typename T>
void rope_(std::byte *out_bytes, const std::byte *in_bytes, const std::byte *pos_ids_bytes,
           float theta, size_t seqlen, size_t nhead, size_t d) {
    auto out = reinterpret_cast<T *>(out_bytes);
    auto in = reinterpret_cast<const T *>(in_bytes);
    auto pos_ids = reinterpret_cast<const int64_t *>(pos_ids_bytes);

    size_t total = seqlen * nhead * d;
    const int blockSize = 256;
    const int numBlocks = (total + blockSize - 1) / blockSize;

    ropeKernel<T><<<numBlocks, blockSize>>>(out, in, pos_ids, theta, seqlen, nhead, d);
}

void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          llaisysDataType_t type, float theta, size_t seqlen, size_t nhead, size_t d) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_<float>(out, in, pos_ids, theta, seqlen, nhead, d);
    case LLAISYS_DTYPE_BF16:
        return rope_<bf16_t_metax>(out, in, pos_ids, theta, seqlen, nhead, d);
    case LLAISYS_DTYPE_F16:
        return rope_<fp16_t_metax>(out, in, pos_ids, theta, seqlen, nhead, d);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

}
