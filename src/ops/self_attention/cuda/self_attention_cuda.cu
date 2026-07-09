#include "self_attention_cuda.cuh"
#include "../../cuda_utils.cuh"

#include <cfloat>

// Optimized self-attention kernel using parallel reduction for dot products.
// Each block handles one (query_pos, head) pair.
// Thread parallelism over key positions for Q*K dot product, then over d for V accumulation.
__global__ void self_attention_kernel(void *attn_val, const void *q, const void *k, const void *v,
                                      float scale, llaisysDataType_t dtype,
                                      size_t qlen, size_t kvlen, size_t nh, size_t nkvh, size_t d) {
    size_t qi = blockIdx.x;
    size_t h = blockIdx.y;
    if (qi >= qlen || h >= nh) return;

    size_t group_size = nh / nkvh;
    size_t kvh = h / group_size;

    extern __shared__ float shared[];
    float *scores = shared;
    float *q_cache = shared + kvlen;
    float *warp_buf = q_cache + d;

    int num_warps = blockDim.x / 32;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    for (size_t di = threadIdx.x; di < d; di += blockDim.x) {
        q_cache[di] = load_as_f32(q, (qi * nh + h) * d + di, dtype);
    }
    __syncthreads();

    size_t max_ki = qi + (kvlen - qlen);

    // Q*K^T: each thread handles multiple key positions
    for (size_t ki = threadIdx.x; ki < kvlen; ki += blockDim.x) {
        if (ki <= max_ki) {
            float dot = 0.0f;
            const size_t k_base = (ki * nkvh + kvh) * d;
            for (size_t di = 0; di < d; di += 4) {
                dot += q_cache[di]     * load_as_f32(k, k_base + di,     dtype);
                dot += q_cache[di + 1] * load_as_f32(k, k_base + di + 1, dtype);
                dot += q_cache[di + 2] * load_as_f32(k, k_base + di + 2, dtype);
                dot += q_cache[di + 3] * load_as_f32(k, k_base + di + 3, dtype);
            }
            scores[ki] = dot * scale;
        } else {
            scores[ki] = -FLT_MAX;
        }
    }
    __syncthreads();

    // Softmax: find max
    float local_max = -FLT_MAX;
    for (size_t ki = threadIdx.x; ki < kvlen; ki += blockDim.x) {
        float s = scores[ki];
        if (s > local_max) local_max = s;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffff, local_max, offset);
        if (other > local_max) local_max = other;
    }
    if (lane_id == 0) warp_buf[warp_id] = local_max;
    __syncthreads();
    if (threadIdx.x < (unsigned)num_warps) local_max = warp_buf[threadIdx.x];
    else local_max = -FLT_MAX;
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffff, local_max, offset);
        if (other > local_max) local_max = other;
    }
    if (threadIdx.x == 0) warp_buf[0] = local_max;
    __syncthreads();
    float max_score = warp_buf[0];

    // Softmax: exp and sum
    float local_sum = 0.0f;
    for (size_t ki = threadIdx.x; ki < kvlen; ki += blockDim.x) {
        float e = expf(scores[ki] - max_score);
        scores[ki] = e;
        local_sum += e;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    if (lane_id == 0) warp_buf[warp_id] = local_sum;
    __syncthreads();
    if (threadIdx.x < (unsigned)num_warps) local_sum = warp_buf[threadIdx.x];
    else local_sum = 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    if (threadIdx.x == 0) warp_buf[0] = 1.0f / local_sum;
    __syncthreads();
    float inv_sum = warp_buf[0];

    for (size_t ki = threadIdx.x; ki < kvlen; ki += blockDim.x) {
        scores[ki] *= inv_sum;
    }
    __syncthreads();

    // Weighted sum of V: each thread handles multiple d dimensions
    for (size_t di = threadIdx.x; di < d; di += blockDim.x) {
        float sum = 0.0f;
        for (size_t ki = 0; ki < kvlen; ki++) {
            sum += scores[ki] * load_as_f32(v, (ki * nkvh + kvh) * d + di, dtype);
        }
        store_from_f32(attn_val, (qi * nh + h) * d + di, sum, dtype);
    }
}

namespace llaisys::ops::cuda {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    float scale, llaisysDataType_t dtype,
                    size_t qlen, size_t kvlen, size_t nh, size_t nkvh, size_t d) {
    int block_size = 256;
    int num_warps = block_size / 32;
    size_t shared_mem = (kvlen + d + num_warps) * sizeof(float);
    dim3 grid(qlen, nh);
    self_attention_kernel<<<grid, block_size, shared_mem>>>(
        attn_val, q, k, v, scale, dtype, qlen, kvlen, nh, nkvh, d);
    CUDA_KERNEL_CHECK();
}
} // namespace llaisys::ops::cuda
