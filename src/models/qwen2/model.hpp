#pragma once

#include "llaisys/models/qwen2.h"
#include "llaisys/tensor.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace llaisys::models {

// Forward-declare internal tensor handle type (defined in src/tensor/tensor.hpp).
// We keep it opaque here to avoid pulling in the full tensor header.
} // namespace llaisys::models

namespace llaisys {
class Tensor;
using tensor_t = std::shared_ptr<Tensor>;
} // namespace llaisys

namespace llaisys::models {

// Skeleton for Assignment #3.
// Keep the model logic here (outside src/llaisys/), and expose it via C API in src/llaisys/models/.
class Qwen2Model {
public:
    static std::unique_ptr<Qwen2Model> create(const LlaisysQwen2Meta &meta,
                                              llaisysDeviceType_t device,
                                              const int *device_ids,
                                              int ndevice);

    ~Qwen2Model();

    // The returned pointer is owned by the model and valid until destroy().
    LlaisysQwen2Weights *weights();

    // Infer one next token given existing token_ids[0..ntoken).
    // NOTE: This is intentionally a stub for the assignment skeleton.
    int64_t infer(const int64_t *token_ids, size_t ntoken);

    int64_t endToken() const { return meta_.end_token; }

private:
    explicit Qwen2Model(const LlaisysQwen2Meta &meta,
                        llaisysDeviceType_t device,
                        std::vector<int> device_ids);

    void initWeightsView_();
    const char *missingWeightsHint_() const;

private:
    LlaisysQwen2Meta meta_{};
    llaisysDeviceType_t device_{LLAISYS_DEVICE_CPU};
    std::vector<int> device_ids_;

    // C-facing view of weights. Pointers inside refer to the vectors below.
    LlaisysQwen2Weights weights_{};

    // Per-layer weight handles (filled by Python via weights()).
    std::vector<llaisysTensor_t> attn_norm_w_;
    std::vector<llaisysTensor_t> attn_q_w_;
    std::vector<llaisysTensor_t> attn_q_b_;
    std::vector<llaisysTensor_t> attn_k_w_;
    std::vector<llaisysTensor_t> attn_k_b_;
    std::vector<llaisysTensor_t> attn_v_w_;
    std::vector<llaisysTensor_t> attn_v_b_;
    std::vector<llaisysTensor_t> attn_o_w_;

    std::vector<llaisysTensor_t> mlp_norm_w_;
    std::vector<llaisysTensor_t> mlp_gate_w_;
    std::vector<llaisysTensor_t> mlp_up_w_;
    std::vector<llaisysTensor_t> mlp_down_w_;

    // TODO(assignment-3): add KV-cache storage here.
    std::vector<::llaisys::tensor_t> k_cache_;
    std::vector<::llaisys::tensor_t> v_cache_;
    size_t cur_len_{0};
};

} // namespace llaisys::models
