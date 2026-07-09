#include "rope_cuda.cuh"

#include "../../../utils.hpp"
#include "../../../utils/check.hpp"

#include <cmath>

namespace llaisys::ops::cuda {

__global__ void rope_kernel(void *out, const void *in, const int64_t *pos_ids, llaisysDataType_t type, size_t seq_len,
                            size_t n_heads, size_t head_dim, float theta) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t half_dim = head_dim / 2;
    size_t total = seq_len * n_heads * half_dim;
    if (idx >= total) {
        return;
    }

    size_t d = idx % half_dim;
    size_t h = (idx / half_dim) % n_heads;
    size_t s = idx / (half_dim * n_heads);

    float pos = static_cast<float>(pos_ids[s]);
    float phi = pos / powf(theta, 2.0f * static_cast<float>(d) / static_cast<float>(head_dim));
    float c = cosf(phi);
    float sn = sinf(phi);

    size_t base = (s * n_heads + h) * head_dim;
    float a = load_as_float(in, base + d, type);
    float b = load_as_float(in, base + half_dim + d, type);

    store_from_float(out, base + d, a * c - b * sn, type);
    store_from_float(out, base + half_dim + d, b * c + a * sn, type);
}

void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, llaisysDataType_t type, size_t seq_len,
          size_t n_heads, size_t head_dim, float theta) {
    ASSERT(type == LLAISYS_DTYPE_F32 || type == LLAISYS_DTYPE_F16 || type == LLAISYS_DTYPE_BF16,
           "rope cuda only supports F32/F16/BF16");
    size_t total = seq_len * n_heads * (head_dim / 2);
    dim3 block(256);
    dim3 grid((total + block.x - 1) / block.x);
    rope_kernel<<<grid, block>>>(out, in, reinterpret_cast<const int64_t *>(pos_ids), type, seq_len, n_heads, head_dim,
                                 theta);
    auto err = cudaGetLastError();
    ASSERT(err == cudaSuccess, "rope cuda kernel launch failed");
}

} // namespace llaisys::ops::cuda
