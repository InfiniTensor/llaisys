#include "sample_cuda.cuh"
#include "../../cuda_utils.cuh"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>
#include <cstdlib>
#include <cstring>

static thread_local std::mt19937 rng{std::random_device{}()};

namespace llaisys::ops::cuda {
void sample(std::byte *out_idx, const std::byte *logits, llaisysDataType_t type, size_t numel,
            float temperature, int top_k, float top_p) {
    // Copy logits from GPU to CPU, do sampling on CPU, copy result back
    size_t esize = cuda_dsize(type);
    std::vector<char> host_logits(numel * esize);
    cudaMemcpy(host_logits.data(), logits, numel * esize, cudaMemcpyDeviceToHost);

    // Convert to float
    std::vector<float> probs(numel);
    for (size_t i = 0; i < numel; i++) {
        if (type == LLAISYS_DTYPE_F32) {
            probs[i] = reinterpret_cast<const float *>(host_logits.data())[i];
        } else if (type == LLAISYS_DTYPE_BF16) {
            uint16_t v = reinterpret_cast<const uint16_t *>(host_logits.data())[i];
            uint32_t bits = static_cast<uint32_t>(v) << 16;
            float f;
            std::memcpy(&f, &bits, sizeof(float));
            probs[i] = f;
        } else if (type == LLAISYS_DTYPE_F16) {
            uint16_t v = reinterpret_cast<const uint16_t *>(host_logits.data())[i];
            // Simple F16 -> F32 conversion
            uint32_t sign = (v >> 15) & 0x1;
            uint32_t exp = (v >> 10) & 0x1F;
            uint32_t mant = v & 0x3FF;
            uint32_t f32_bits;
            if (exp == 0) {
                f32_bits = sign << 31;
            } else if (exp == 0x1F) {
                f32_bits = (sign << 31) | 0x7F800000 | (mant << 13);
            } else {
                f32_bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
            }
            float f;
            std::memcpy(&f, &f32_bits, sizeof(float));
            probs[i] = f;
        }
    }

    if (temperature <= 0.0f) temperature = 1.0f;
    if (temperature != 1.0f) {
        for (size_t i = 0; i < numel; i++) {
            probs[i] /= temperature;
        }
    }

    std::vector<int> indices(numel);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&](int a, int b) { return probs[a] > probs[b]; });

    size_t keep = numel;
    if (top_k > 0 && static_cast<size_t>(top_k) < numel) {
        keep = static_cast<size_t>(top_k);
    }

    float max_val = probs[indices[0]];
    std::vector<float> softmax_vals(keep);
    float sum_exp = 0.0f;
    for (size_t i = 0; i < keep; i++) {
        softmax_vals[i] = std::exp(probs[indices[i]] - max_val);
        sum_exp += softmax_vals[i];
    }
    for (size_t i = 0; i < keep; i++) {
        softmax_vals[i] /= sum_exp;
    }

    if (top_p > 0.0f && top_p < 1.0f) {
        float cumsum = 0.0f;
        size_t cutoff = keep;
        for (size_t i = 0; i < keep; i++) {
            cumsum += softmax_vals[i];
            if (cumsum >= top_p) {
                cutoff = i + 1;
                break;
            }
        }
        keep = cutoff;
        float new_sum = 0.0f;
        for (size_t i = 0; i < keep; i++) new_sum += softmax_vals[i];
        for (size_t i = 0; i < keep; i++) softmax_vals[i] /= new_sum;
    }

    std::discrete_distribution<int> dist(softmax_vals.begin(), softmax_vals.begin() + keep);
    int sampled = dist(rng);
    int64_t result = static_cast<int64_t>(indices[sampled]);

    cudaMemcpy(out_idx, &result, sizeof(int64_t), cudaMemcpyHostToDevice);
}
} // namespace llaisys::ops::cuda
