#include "random_sample_nvidia.hpp"

#include "../../argmax/nvidia/argmax_nvidia.hpp"
#include "../../nvidia_cuda.cuh"
#include "../cpu/random_sample_cpu.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void cast_scale_kernel(float *out, const T *in, size_t n, float inv_temperature) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    float v = to_float(in[idx]);
    if (!isfinite(v)) {
        v = v > 0.0f ? 1e30f : -1e30f;
    }
    out[idx] = v * inv_temperature;
}

template <typename T>
void launch_cast_scale(float *out, const std::byte *in, size_t n, float inv_temperature) {
    const int threads = num_threads_1d();
    const int blocks = num_blocks_1d(n, threads);
    auto stream = current_stream();
    cast_scale_kernel<<<blocks, threads, 0, stream>>>(
        out,
        reinterpret_cast<const T *>(in),
        n,
        inv_temperature);
    check_cuda(cudaGetLastError(), "random_sample cast_scale kernel launch");
}

__global__ void mask_one_kernel(float *vals, int64_t idx, size_t n) {
    if (threadIdx.x == 0 && blockIdx.x == 0 && idx >= 0 && static_cast<size_t>(idx) < n) {
        vals[idx] = -1e30f;
    }
}

static int64_t sample_from_sorted_host(
    const std::vector<float> &sorted_logits,
    const std::vector<int64_t> &sorted_indices,
    float top_p,
    uint64_t seed) {
    if (sorted_logits.empty() || sorted_indices.empty()) {
        return 0;
    }

    if (!std::isfinite(top_p)) {
        top_p = 1.0f;
    }
    if (top_p < 0.0f) {
        top_p = 0.0f;
    }
    if (top_p > 1.0f) {
        top_p = 1.0f;
    }

    if (top_p <= 0.0f || sorted_logits.size() == 1) {
        return sorted_indices[0];
    }

    const float max_logit = sorted_logits[0];
    std::vector<double> weights(sorted_logits.size(), 0.0);
    double weight_sum = 0.0;
    for (size_t i = 0; i < sorted_logits.size(); ++i) {
        const double w = std::exp(static_cast<double>(sorted_logits[i] - max_logit));
        weights[i] = w;
        weight_sum += w;
    }

    if (!(weight_sum > 0.0) || !std::isfinite(weight_sum)) {
        return sorted_indices[0];
    }

    size_t keep = sorted_logits.size();
    if (top_p < 1.0f) {
        double cumulative = 0.0;
        keep = 0;
        for (size_t i = 0; i < weights.size(); ++i) {
            cumulative += weights[i] / weight_sum;
            keep = i + 1;
            if (cumulative >= static_cast<double>(top_p)) {
                break;
            }
        }
        if (keep == 0) {
            keep = 1;
        }
    }

    std::mt19937_64 rng(seed);
    std::discrete_distribution<size_t> dist(weights.begin(), weights.begin() + static_cast<ptrdiff_t>(keep));
    return sorted_indices[dist(rng)];
}

void random_sample(
    std::byte *out_idx,
    const std::byte *logits,
    llaisysDataType_t type,
    size_t numel,
    float temperature,
    size_t top_k,
    float top_p,
    uint64_t seed) {
    ASSERT(out_idx != nullptr, "RandomSample(nvidia): out_idx cannot be null.");
    ASSERT(logits != nullptr, "RandomSample(nvidia): logits cannot be null.");
    ASSERT(numel > 0, "RandomSample(nvidia): numel must be positive.");

    const bool deterministic = !std::isfinite(temperature) || temperature <= 0.0f || top_k == 1 || top_p <= 0.0f;
    const float inv_temperature = deterministic ? 1.0f : (1.0f / temperature);

    float *d_scaled = nullptr;
    int64_t *d_max_idx = nullptr;
    float *d_max_val = nullptr;

    check_cuda(cudaMalloc(&d_scaled, numel * sizeof(float)), "random_sample cudaMalloc scaled");
    check_cuda(cudaMalloc(&d_max_idx, sizeof(int64_t)), "random_sample cudaMalloc max_idx");
    check_cuda(cudaMalloc(&d_max_val, sizeof(float)), "random_sample cudaMalloc max_val");

    switch (type) {
    case LLAISYS_DTYPE_F32:
        launch_cast_scale<float>(d_scaled, logits, numel, inv_temperature);
        break;
    case LLAISYS_DTYPE_F16:
        launch_cast_scale<llaisys::fp16_t>(d_scaled, logits, numel, inv_temperature);
        break;
    case LLAISYS_DTYPE_BF16:
        launch_cast_scale<llaisys::bf16_t>(d_scaled, logits, numel, inv_temperature);
        break;
    default:
        cudaFree(d_scaled);
        cudaFree(d_max_idx);
        cudaFree(d_max_val);
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    auto stream = current_stream();
    int64_t sampled_idx = 0;

    if (deterministic) {
        llaisys::ops::nvidia::argmax(
            reinterpret_cast<std::byte *>(d_max_idx),
            reinterpret_cast<std::byte *>(d_max_val),
            reinterpret_cast<const std::byte *>(d_scaled),
            LLAISYS_DTYPE_F32,
            numel);
        check_cuda(
            cudaMemcpyAsync(&sampled_idx, d_max_idx, sizeof(sampled_idx), cudaMemcpyDeviceToHost, stream),
            "random_sample copy deterministic idx");
        check_cuda(cudaStreamSynchronize(stream), "random_sample sync deterministic idx");
        check_cuda(
            cudaMemcpy(out_idx, &sampled_idx, sizeof(sampled_idx), cudaMemcpyHostToDevice),
            "random_sample deterministic copy idx");
        cudaFree(d_scaled);
        cudaFree(d_max_idx);
        cudaFree(d_max_val);
        return;
    }

    size_t k = top_k;
    if (k == 0 || k > numel) {
        k = numel;
    }

    // Avoid O(k * vocab) kernel loops for very large k by falling back to single D2H copy.
    constexpr size_t kIterativeTopKLimit = 1024;
    if (k > kIterativeTopKLimit) {
        std::vector<float> host_scaled(numel);
        check_cuda(
            cudaMemcpyAsync(host_scaled.data(), d_scaled, numel * sizeof(float), cudaMemcpyDeviceToHost, stream),
            "random_sample fallback copy scaled");
        check_cuda(cudaStreamSynchronize(stream), "random_sample fallback sync copy scaled");
        cpu::random_sample(
            reinterpret_cast<std::byte *>(&sampled_idx),
            reinterpret_cast<const std::byte *>(host_scaled.data()),
            LLAISYS_DTYPE_F32,
            numel,
            1.0f,
            top_k,
            top_p,
            seed);
    } else {
        std::vector<float> top_logits;
        std::vector<int64_t> top_indices;
        top_logits.reserve(k);
        top_indices.reserve(k);

        for (size_t i = 0; i < k; ++i) {
            float h_max_val = -1e30f;
            int64_t h_max_idx = 0;

            llaisys::ops::nvidia::argmax(
                reinterpret_cast<std::byte *>(d_max_idx),
                reinterpret_cast<std::byte *>(d_max_val),
                reinterpret_cast<const std::byte *>(d_scaled),
                LLAISYS_DTYPE_F32,
                numel);

            check_cuda(
                cudaMemcpyAsync(&h_max_idx, d_max_idx, sizeof(h_max_idx), cudaMemcpyDeviceToHost, stream),
                "random_sample copy topk idx");
            check_cuda(
                cudaMemcpyAsync(&h_max_val, d_max_val, sizeof(h_max_val), cudaMemcpyDeviceToHost, stream),
                "random_sample copy topk val");
            check_cuda(cudaStreamSynchronize(stream), "random_sample sync topk pair");

            top_indices.push_back(h_max_idx);
            top_logits.push_back(h_max_val);

            mask_one_kernel<<<1, 1, 0, stream>>>(d_scaled, h_max_idx, numel);
            check_cuda(cudaGetLastError(), "random_sample mask_one kernel launch");
        }

        sampled_idx = sample_from_sorted_host(top_logits, top_indices, top_p, seed);
    }

    check_cuda(
        cudaMemcpy(out_idx, &sampled_idx, sizeof(sampled_idx), cudaMemcpyHostToDevice),
        "random_sample sampled copy idx");

    cudaFree(d_scaled);
    cudaFree(d_max_idx);
    cudaFree(d_max_val);
}

} // namespace llaisys::ops::nvidia
