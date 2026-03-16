#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace {

constexpr float NEG_INF = -1e10f;

template <typename T>
void logits_to_float(std::vector<float> &out, const T *logits, size_t n) {
    out.resize(n);
    for (size_t i = 0; i < n; i++) {
        if constexpr (std::is_same_v<T, llaisys::fp16_t> ||
                      std::is_same_v<T, llaisys::bf16_t>) {
            out[i] = llaisys::utils::cast<float>(logits[i]);
        } else {
            out[i] = static_cast<float>(logits[i]);
        }
    }
}

void sample_cpu_impl(std::vector<float> &logits_f, int top_k, float top_p,
                    uint64_t seed, int64_t *out_idx) {
    const size_t n = logits_f.size();
    if (n == 0) {
        *out_idx = 0;
        return;
    }

    // Top-K: 只保留 logit 值最大的 k 个，其余置为 -inf
    if (top_k > 0 && static_cast<size_t>(top_k) < n) {
        std::vector<float> copy = logits_f;
        std::nth_element(copy.begin(), copy.begin() + top_k - 1, copy.end(),
                         std::greater<float>());
        float thresh = copy[top_k - 1];
        for (size_t i = 0; i < n; i++) {
            if (logits_f[i] < thresh) logits_f[i] = NEG_INF;
        }
    }

    // Softmax（-inf 会变成 0）
    float max_logit = *std::max_element(logits_f.begin(), logits_f.end());
    if (max_logit == NEG_INF) {
        *out_idx = 0;
        return;
    }
    double sum = 0;
    for (size_t i = 0; i < n; i++) {
        float x = logits_f[i];
        logits_f[i] = (x == NEG_INF) ? 0.f
                                    : static_cast<float>(std::exp(static_cast<double>(x - max_logit)));
        sum += static_cast<double>(logits_f[i]);
    }
    if (sum <= 0) {
        *out_idx = 0;
        return;
    }
    for (size_t i = 0; i < n; i++)
        logits_f[i] = static_cast<float>(static_cast<double>(logits_f[i]) / sum);

    // Top-P (nucleus): 按概率从高到低排序，只保留累积概率达到 top_p 的前缀，其余置 0 再归一化
    if (top_p > 0.f && top_p < 1.f) {
        std::vector<float> probs = logits_f;
        std::vector<size_t> idx(n);
        for (size_t i = 0; i < n; i++) idx[i] = i;
        std::sort(idx.begin(), idx.end(),
                  [&probs](size_t a, size_t b) { return probs[a] > probs[b]; });
        float cum = 0.f;
        size_t cut = n;
        for (size_t i = 0; i < n; i++) {
            cum += probs[idx[i]];
            if (cum >= top_p) {
                cut = i + 1;
                break;
            }
        }
        for (size_t i = 0; i < n; i++) logits_f[i] = 0.f;
        for (size_t i = 0; i < cut; i++)
            logits_f[idx[i]] = probs[idx[i]];
        double sum2 = 0;
        for (size_t i = 0; i < n; i++) sum2 += logits_f[i];
        if (sum2 > 0)
            for (size_t i = 0; i < n; i++)
                logits_f[i] = static_cast<float>(logits_f[i] / sum2);
    }

    // 多项式采样
    std::mt19937 rng(seed != 0 ? static_cast<std::mt19937::result_type>(seed)
                               : static_cast<std::mt19937::result_type>(
                                     std::random_device{}()));
    std::uniform_real_distribution<float> u(0.f, 1.f);
    float r = u(rng);
    float cum = 0.f;
    for (size_t i = 0; i < n; i++) {
        cum += logits_f[i];
        if (r <= cum) {
            *out_idx = static_cast<int64_t>(i);
            return;
        }
    }
    *out_idx = static_cast<int64_t>(n - 1);
}

void sample_cpu(std::byte *out_idx, const std::byte *logits,
                llaisysDataType_t logits_type, size_t numel, float temperature,
                int top_k, float top_p, uint64_t seed) {
    std::vector<float> logits_f;
    switch (logits_type) {
    case LLAISYS_DTYPE_F32:
        logits_to_float(logits_f, reinterpret_cast<const float *>(logits), numel);
        break;
    case LLAISYS_DTYPE_F16:
        logits_to_float(logits_f,
                        reinterpret_cast<const llaisys::fp16_t *>(logits), numel);
        break;
    case LLAISYS_DTYPE_BF16:
        logits_to_float(logits_f,
                        reinterpret_cast<const llaisys::bf16_t *>(logits), numel);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(logits_type);
    }

    // Temperature：极小或 <=0 时退化为 argmax
    if (temperature <= 0.f || temperature < 1e-6f) {
        size_t best = 0;
        for (size_t i = 1; i < logits_f.size(); i++) {
            if (logits_f[i] > logits_f[best]) best = i;
        }
        *reinterpret_cast<int64_t *>(out_idx) = static_cast<int64_t>(best);
        return;
    }
    for (auto &v : logits_f) v /= temperature;

    sample_cpu_impl(logits_f, top_k, top_p, seed,
                    reinterpret_cast<int64_t *>(out_idx));
}

} // namespace

namespace llaisys::ops {

void sample(tensor_t out_idx, tensor_t logits, float temperature, int top_k,
            float top_p, uint64_t seed) {
    CHECK_SAME_DEVICE(out_idx, logits);
    ASSERT(out_idx->dtype() == LLAISYS_DTYPE_I64, "sample: out_idx must be int64");
    ASSERT(logits->isContiguous() && out_idx->isContiguous(),
           "sample: tensors must be contiguous");
    ASSERT(logits->ndim() == 1, "sample: logits must be 1D");
    ASSERT(out_idx->numel() == 1, "sample: out_idx must have one element");

    if (logits->deviceType() == LLAISYS_DEVICE_CPU) {
        return sample_cpu(out_idx->data(), logits->data(), logits->dtype(),
                          logits->numel(), temperature, top_k, top_p, seed);
    }

    llaisys::core::context().setDevice(logits->deviceType(), logits->deviceId());
    switch (logits->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return sample_cpu(out_idx->data(), logits->data(), logits->dtype(),
                          logits->numel(), temperature, top_k, top_p, seed);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA: {
        size_t logits_bytes = logits->numel() * logits->elementSize();
        std::vector<std::byte> host_logits(logits_bytes);
        llaisys::core::context().runtime().api()->memcpy_sync(host_logits.data(), logits->data(), logits_bytes, LLAISYS_MEMCPY_D2H);
        int64_t host_idx;
        sample_cpu(reinterpret_cast<std::byte *>(&host_idx), host_logits.data(), logits->dtype(), logits->numel(), temperature, top_k, top_p, seed);
        llaisys::core::context().runtime().api()->memcpy_sync(out_idx->data(), &host_idx, sizeof(int64_t), LLAISYS_MEMCPY_H2D);
        return;
    }
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace llaisys::ops
