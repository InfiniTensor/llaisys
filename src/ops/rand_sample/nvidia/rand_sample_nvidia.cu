#include "rand_sample_nvidia.hpp"

#include "../../../core/llaisys_core.hpp"
#include "../../nvidia_utils.cuh"

#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace llaisys::ops::nvidia {
__device__ inline uint64_t lcg_next(uint64_t &state) {
    state = state * 6364136223846793005ULL + 1ULL;
    return state;
}

__device__ inline float rng_uniform(uint64_t &state) {
    uint64_t x = lcg_next(state);
    uint32_t mant = static_cast<uint32_t>(x >> 40);
    return static_cast<float>(mant) * (1.0f / 16777216.0f);
}

template <typename T>
__global__ void rand_sample_kernel(int64_t *out_idx, T *out_val, const T *vals, size_t numel, int64_t batch_size,
                                   float temperature, size_t topK, float topP, int64_t seed, float *probs, int *idx) {
    int64_t b = static_cast<int64_t>(blockIdx.x);
    if (b >= batch_size) {
        return;
    }
    if (threadIdx.x != 0) {
        return;
    }
    const T *v = vals + static_cast<size_t>(b) * numel;
    float *p = probs + static_cast<size_t>(b) * numel;
    int *id = idx + static_cast<size_t>(b) * numel;
    float temp = temperature <= 1e-6f ? 1e-6f : temperature;
    size_t max_i = 0;
    float max_v = detail::to_float(v[0]);
    for (size_t i = 1; i < numel; ++i) {
        float cur = detail::to_float(v[i]);
        if (cur > max_v) {
            max_v = cur;
            max_i = i;
        }
    }
    float sum = 0.0f;
    for (size_t i = 0; i < numel; ++i) {
        float cur = detail::to_float(v[i]);
        float val = expf((cur - max_v) / temp);
        p[i] = val;
        sum += val;
        id[i] = static_cast<int>(i);
    }
    if (sum <= 0.0f) {
        for (size_t i = 0; i < numel; ++i) {
            p[i] = 0.0f;
        }
        p[max_i] = 1.0f;
        sum = 1.0f;
    } else {
        float inv = 1.0f / sum;
        for (size_t i = 0; i < numel; ++i) {
            p[i] *= inv;
        }
    }
    size_t k = numel;
    if (topK > 0 && topK < numel) {
        k = topK;
    }
    if ((topK > 0 && topK < numel) || (topP > 0.0f && topP < 1.0f)) {
        for (size_t i = 0; i < k; ++i) {
            size_t max_pos = i;
            float max_val = p[i];
            for (size_t j = i + 1; j < numel; ++j) {
                float vj = p[j];
                if (vj > max_val) {
                    max_val = vj;
                    max_pos = j;
                }
            }
            if (max_pos != i) {
                float tmp = p[i];
                p[i] = p[max_pos];
                p[max_pos] = tmp;
                int tmp_i = id[i];
                id[i] = id[max_pos];
                id[max_pos] = tmp_i;
            }
        }
    }
    size_t cand = k;
    if (topP > 0.0f && topP < 1.0f) {
        float cum = 0.0f;
        cand = 0;
        for (size_t i = 0; i < k; ++i) {
            cum += p[i];
            cand = i + 1;
            if (cum >= topP) {
                break;
            }
        }
        if (cand == 0) {
            cand = 1;
        }
    }
    float cand_sum = 0.0f;
    for (size_t i = 0; i < cand; ++i) {
        cand_sum += p[i];
    }
    size_t chosen = static_cast<size_t>(id[0]);
    if (cand_sum > 0.0f) {
        uint64_t state = static_cast<uint64_t>(seed) ^ (static_cast<uint64_t>(b + 1) * 0x9e3779b97f4a7c15ULL);
        float r = rng_uniform(state) * cand_sum;
        for (size_t i = 0; i < cand; ++i) {
            r -= p[i];
            if (r <= 0.0f) {
                chosen = static_cast<size_t>(id[i]);
                break;
            }
            if (i == cand - 1) {
                chosen = static_cast<size_t>(id[i]);
            }
        }
    }
    out_idx[b] = static_cast<int64_t>(chosen);
    out_val[b] = v[chosen];
}

void rand_sample(std::byte *sample_idx, std::byte *sample_val, const std::byte *vals, llaisysDataType_t type, size_t numel,
                 int64_t batch_size, float temperature, size_t topK, float topP, int64_t seed) {
    auto &runtime = llaisys::core::context().runtime();
    if (batch_size <= 0 || numel == 0) {
        return;
    }
    auto d_probs = static_cast<float *>(runtime.api()->malloc_device(sizeof(float) * static_cast<size_t>(batch_size) * numel));
    auto d_idx = static_cast<int *>(runtime.api()->malloc_device(sizeof(int) * static_cast<size_t>(batch_size) * numel));
    dim3 grid(static_cast<unsigned int>(batch_size), 1, 1);
    int threads = 1;
    auto stream = reinterpret_cast<cudaStream_t>(runtime.stream());
    switch (type) {
    case LLAISYS_DTYPE_F32:
        rand_sample_kernel<<<grid, threads, 0, stream>>>(
            reinterpret_cast<int64_t *>(sample_idx),
            reinterpret_cast<float *>(sample_val),
            reinterpret_cast<const float *>(vals),
            numel, batch_size, temperature, topK, topP, seed, d_probs, d_idx);
        break;
    case LLAISYS_DTYPE_BF16:
        rand_sample_kernel<<<grid, threads, 0, stream>>>(
            reinterpret_cast<int64_t *>(sample_idx),
            reinterpret_cast<llaisys::bf16_t *>(sample_val),
            reinterpret_cast<const llaisys::bf16_t *>(vals),
            numel, batch_size, temperature, topK, topP, seed, d_probs, d_idx);
        break;
    case LLAISYS_DTYPE_F16:
        rand_sample_kernel<<<grid, threads, 0, stream>>>(
            reinterpret_cast<int64_t *>(sample_idx),
            reinterpret_cast<llaisys::fp16_t *>(sample_val),
            reinterpret_cast<const llaisys::fp16_t *>(vals),
            numel, batch_size, temperature, topK, topP, seed, d_probs, d_idx);
        break;
    default:
        runtime.api()->free_device(d_probs);
        runtime.api()->free_device(d_idx);
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
    detail::checkCuda(cudaGetLastError());
    detail::checkCuda(cudaStreamSynchronize(stream));
    runtime.api()->free_device(d_probs);
    runtime.api()->free_device(d_idx);
}
}
