#include "rope_cuda.cuh"
#include "../../cuda_utils.cuh"

// Each thread handles one (seq, head, pair_idx) triple
__global__ void rope_kernel(void *out, const void *in, const int64_t *pos_ids,
                            float theta, llaisysDataType_t dtype,
                            size_t seqlen, size_t nhead, size_t d) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t half_d = d / 2;
    size_t total = seqlen * nhead * half_d;
    if (idx >= total) return;

    size_t j = idx % half_d;
    size_t h = (idx / half_d) % nhead;
    size_t s = idx / (half_d * nhead);

    float pos = static_cast<float>(pos_ids[s]);
    float theta_pow = powf(theta, 2.0f * static_cast<float>(j) / static_cast<float>(d));
    float phi = pos / theta_pow;
    float cos_phi = cosf(phi);
    float sin_phi = sinf(phi);

    size_t base = (s * nhead + h) * d;
    float a_val = load_as_f32(in, base + j, dtype);
    float b_val = load_as_f32(in, base + half_d + j, dtype);

    store_from_f32(out, base + j, a_val * cos_phi - b_val * sin_phi, dtype);
    store_from_f32(out, base + half_d + j, b_val * cos_phi + a_val * sin_phi, dtype);
}

namespace llaisys::ops::cuda {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          float theta, llaisysDataType_t dtype,
          size_t seqlen, size_t nhead, size_t d) {
    size_t total = seqlen * nhead * (d / 2);
    rope_kernel<<<cuda_grid_size(total), CUDA_BLOCK_SIZE>>>(
        out, in, reinterpret_cast<const int64_t *>(pos_ids),
        theta, dtype, seqlen, nhead, d);
    CUDA_KERNEL_CHECK();
}
} // namespace llaisys::ops::cuda
