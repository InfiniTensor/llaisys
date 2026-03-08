#include "op.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <numeric>
#include <utility>
#include <vector>

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

namespace {

template <typename T>
void load_logits(std::vector<float> &dst, const T *src, size_t n) {
    dst.resize(n);
    for (size_t i = 0; i < n; ++i) {
        dst[i] = llaisys::utils::cast<float>(src[i]);
    }
}

int64_t argmax_index(const std::vector<float> &vals) {
    return static_cast<int64_t>(
        std::distance(vals.begin(), std::max_element(vals.begin(), vals.end())));
}

int64_t sample_from_logits(std::vector<float> logits, float temperature, int top_k, float top_p) {
    if (logits.empty()) {
        return 0;
    }

    if (temperature <= 0.0f) {
        return argmax_index(logits);
    }

    for (auto &v : logits) {
        v /= temperature;
    }

    float max_logit = *std::max_element(logits.begin(), logits.end());
    std::vector<float> probs(logits.size(), 0.0f);
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum += probs[i];
    }
    if (sum <= 0.0f) {
        return argmax_index(logits);
    }
    for (auto &p : probs) {
        p /= sum;
    }

    std::vector<size_t> keep_idx(probs.size());
    std::iota(keep_idx.begin(), keep_idx.end(), 0);
    std::sort(keep_idx.begin(), keep_idx.end(), [&probs](size_t a, size_t b) {
        return probs[a] > probs[b];
    });

    if (top_k > 0 && static_cast<size_t>(top_k) < keep_idx.size()) {
        keep_idx.resize(static_cast<size_t>(top_k));
    }

    if (top_p > 0.0f && top_p < 1.0f && !keep_idx.empty()) {
        std::vector<size_t> nucleus;
        nucleus.reserve(keep_idx.size());
        float cumulative = 0.0f;
        for (size_t idx : keep_idx) {
            nucleus.push_back(idx);
            cumulative += probs[idx];
            if (cumulative >= top_p) {
                break;
            }
        }
        keep_idx.swap(nucleus);
    }

    if (keep_idx.empty()) {
        return argmax_index(logits);
    }

    std::vector<float> filtered;
    filtered.reserve(keep_idx.size());
    for (size_t idx : keep_idx) {
        filtered.push_back(probs[idx]);
    }

    float filtered_sum = std::accumulate(filtered.begin(), filtered.end(), 0.0f);
    if (filtered_sum <= 0.0f) {
        return static_cast<int64_t>(keep_idx[0]);
    }
    for (auto &v : filtered) {
        v /= filtered_sum;
    }

    static bool seeded = false;
    if (!seeded) {
        std::srand(static_cast<unsigned int>(std::time(nullptr)));
        seeded = true;
    }

    float r = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    float cdf = 0.0f;
    size_t chosen = filtered.size() - 1;
    for (size_t i = 0; i < filtered.size(); ++i) {
        cdf += filtered[i];
        if (r <= cdf) {
            chosen = i;
            break;
        }
    }
    return static_cast<int64_t>(keep_idx[chosen]);
}

template <typename T>
int64_t sample_impl(const T *logits, size_t n, float temperature, int top_k, float top_p) {
    std::vector<float> host_logits;
    load_logits(host_logits, logits, n);
    return sample_from_logits(std::move(host_logits), temperature, top_k, top_p);
}

} // namespace

namespace llaisys::ops {

void sample(tensor_t out_idx, tensor_t logits, float temperature, int top_k, float top_p) {
    CHECK_SAME_DEVICE(out_idx, logits);
    ASSERT(out_idx->isContiguous() && logits->isContiguous(),
           "Sample: out_idx and logits must be contiguous.");
    CHECK_ARGUMENT(out_idx->dtype() == LLAISYS_DTYPE_I64, "Sample: out_idx must be int64.");
    CHECK_ARGUMENT(out_idx->numel() == 1, "Sample: out_idx must contain exactly one element.");
    CHECK_ARGUMENT(logits->ndim() == 1, "Sample: logits must be 1D.");
    CHECK_ARGUMENT(logits->numel() > 0, "Sample: logits must not be empty.");
    CHECK_ARGUMENT(top_k >= 0, "Sample: top_k must be >= 0.");
    CHECK_ARGUMENT(top_p >= 0.0f && top_p <= 1.0f, "Sample: top_p must be in [0, 1].");

    auto type = logits->dtype();
    size_t n = logits->numel();

    if (logits->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        auto &ctx = llaisys::core::context();
        ctx.setDevice(logits->deviceType(), logits->deviceId());
        const auto *api = ctx.runtime().api();

        const size_t logits_bytes = logits->numel() * logits->elementSize();
        std::vector<std::byte> h_logits(logits_bytes);
        api->memcpy_sync(h_logits.data(), logits->data(), logits_bytes, LLAISYS_MEMCPY_D2H);

        int64_t sampled = 0;
        switch (type) {
        case LLAISYS_DTYPE_F32:
            sampled = sample_impl(reinterpret_cast<const float *>(h_logits.data()), n, temperature, top_k, top_p);
            break;
        case LLAISYS_DTYPE_F16:
            sampled = sample_impl(reinterpret_cast<const llaisys::fp16_t *>(h_logits.data()), n, temperature, top_k, top_p);
            break;
        case LLAISYS_DTYPE_BF16:
            sampled = sample_impl(reinterpret_cast<const llaisys::bf16_t *>(h_logits.data()), n, temperature, top_k, top_p);
            break;
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
        }

        api->memcpy_sync(out_idx->data(), &sampled, sizeof(int64_t), LLAISYS_MEMCPY_H2D);
        return;
    }

    if (logits->deviceType() != LLAISYS_DEVICE_CPU) {
        EXCEPTION_UNSUPPORTED_DEVICE;
    }

    auto *idx_ptr = reinterpret_cast<int64_t *>(out_idx->data());
    switch (type) {
    case LLAISYS_DTYPE_F32:
        *idx_ptr = sample_impl(reinterpret_cast<const float *>(logits->data()), n, temperature, top_k, top_p);
        return;
    case LLAISYS_DTYPE_F16:
        *idx_ptr = sample_impl(reinterpret_cast<const llaisys::fp16_t *>(logits->data()), n, temperature, top_k, top_p);
        return;
    case LLAISYS_DTYPE_BF16:
        *idx_ptr = sample_impl(reinterpret_cast<const llaisys::bf16_t *>(logits->data()), n, temperature, top_k, top_p);
        return;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops
