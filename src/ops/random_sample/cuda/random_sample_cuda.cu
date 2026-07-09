#include "random_sample_cuda.cuh"

#include "../../../utils.hpp"
#include "../../../utils/check.hpp"

#include <cmath>
#include <cfloat>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <curand_kernel.h>
#include <atomic>
#include <chrono>

namespace llaisys::ops::cuda {

__global__ void init_indices_kernel(int64_t *indices, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    indices[idx] = static_cast<int64_t>(idx);
}

__global__ void scale_logits_kernel(float *scaled, const void *logits, llaisysDataType_t type, size_t vocab_size,
                                    float inv_temperature) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vocab_size) {
        return;
    }
    float x = load_as_float(logits, idx, type);
    scaled[idx] = x * inv_temperature;
}

__global__ void random_sample_kernel(int64_t *sample_idx, void *sample_val, const void *logits,
                                     const float *sorted_scaled_logits, const int64_t *sorted_indices,
                                     llaisysDataType_t type, size_t top_k, float top_p,
                                     unsigned long long seed) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    if (top_k == 0) {
        sample_idx[0] = 0;
        store_from_float(sample_val, 0, load_as_float(logits, 0, type), type);
        return;
    }

    top_p = fminf(1.0f, fmaxf(0.0f, top_p));
    if (top_p <= 0.0f) {
        top_k = 1;
        top_p = 1.0f;
    }

    const float max_logit = sorted_scaled_logits[0];
    float sum_exp = 0.0f;
    for (size_t i = 0; i < top_k; ++i) {
        sum_exp += expf(sorted_scaled_logits[i] - max_logit);
    }

    size_t keep_count = top_k;
    if (top_p < 1.0f) {
        float cumulative = 0.0f;
        keep_count = 0;
        for (size_t i = 0; i < top_k; ++i) {
            float p = expf(sorted_scaled_logits[i] - max_logit) / sum_exp;
            cumulative += p;
            keep_count = i + 1;
            if (cumulative >= top_p) {
                break;
            }
        }
        if (keep_count == 0) {
            keep_count = 1;
        }
    }

    float renorm = 0.0f;
    for (size_t i = 0; i < keep_count; ++i) {
        renorm += expf(sorted_scaled_logits[i] - max_logit) / sum_exp;
    }

    curandStatePhilox4_32_10_t rng_state;
    curand_init(seed, 0ULL, 0ULL, &rng_state);
    float r = curand_uniform(&rng_state);

    float running = 0.0f;
    size_t sampled_i = keep_count - 1;
    for (size_t i = 0; i < keep_count; ++i) {
        float p = (expf(sorted_scaled_logits[i] - max_logit) / sum_exp) / renorm;
        running += p;
        if (r <= running) {
            sampled_i = i;
            break;
        }
    }

    const int64_t sampled_vocab_idx = sorted_indices[sampled_i];
    sample_idx[0] = sampled_vocab_idx;
    const float sampled_val = load_as_float(logits, static_cast<size_t>(sampled_vocab_idx), type);
    store_from_float(sample_val, 0, sampled_val, type);
}

void random_sample(std::byte *sample_idx, std::byte *sample_val, const std::byte *logits, llaisysDataType_t type,
                   size_t vocab_size, float temperature, int top_k, float top_p, int64_t seed) {
    static std::atomic<unsigned long long> seed_counter{0};
    unsigned long long sample_seed = 0;
    if (seed >= 0) {
        sample_seed = static_cast<unsigned long long>(seed);
    } else {
        sample_seed =
            static_cast<unsigned long long>(
                std::chrono::high_resolution_clock::now().time_since_epoch().count()) ^
            (seed_counter.fetch_add(1, std::memory_order_relaxed) + 0x9e3779b97f4a7c15ULL);
    }

    ASSERT(type == LLAISYS_DTYPE_F32 || type == LLAISYS_DTYPE_F16 || type == LLAISYS_DTYPE_BF16,
           "random_sample cuda only supports F32/F16/BF16");

    const float inv_temperature = 1.0f / fmaxf(temperature, 1e-6f);
    size_t k = (top_k <= 0 || static_cast<size_t>(top_k) > vocab_size) ? vocab_size : static_cast<size_t>(top_k);

    float *d_scaled = nullptr;
    int64_t *d_indices = nullptr;
    auto err = cudaMalloc(&d_scaled, sizeof(float) * vocab_size);
    ASSERT(err == cudaSuccess, "random_sample cuda malloc d_scaled failed");
    err = cudaMalloc(&d_indices, sizeof(int64_t) * vocab_size);
    ASSERT(err == cudaSuccess, "random_sample cuda malloc d_indices failed");

    dim3 block(256);
    dim3 grid((vocab_size + block.x - 1) / block.x);
    init_indices_kernel<<<grid, block>>>(d_indices, vocab_size);
    err = cudaGetLastError();
    ASSERT(err == cudaSuccess, "random_sample init_indices kernel launch failed");

    scale_logits_kernel<<<grid, block>>>(d_scaled, logits, type, vocab_size, inv_temperature);
    err = cudaGetLastError();
    ASSERT(err == cudaSuccess, "random_sample scale_logits kernel launch failed");

    thrust::device_ptr<float> scaled_ptr(d_scaled);
    thrust::device_ptr<int64_t> indices_ptr(d_indices);
    thrust::sort_by_key(thrust::device, scaled_ptr, scaled_ptr + vocab_size, indices_ptr, thrust::greater<float>());

    random_sample_kernel<<<1, 1>>>(reinterpret_cast<int64_t *>(sample_idx), sample_val, logits, d_scaled, d_indices,
                                   type, k, top_p, sample_seed);
    err = cudaGetLastError();
    ASSERT(err == cudaSuccess, "random_sample sample kernel launch failed");

    cudaFree(d_scaled);
    cudaFree(d_indices);
}

} // namespace llaisys::ops::cuda
