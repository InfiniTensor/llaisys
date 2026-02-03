#pragma once

#include "llaisys/models/qwen2.h"
#include "llaisys/tensor.h"
#include "../transformer/decoder/decoder.hpp"

#include <random>
#include <vector>

namespace llaisys::models {
class Qwen2 {
public:
    Qwen2(const LlaisysQwen2Meta &meta,
          const LlaisysQwen2Weights &weights,
          llaisysDeviceType_t device,
          const std::vector<int> &device_ids);
    ~Qwen2();

    // Compatibility entrypoint; prefer prefill/step for streaming.
    int64_t infer(const int64_t *token_ids, size_t ntoken);
    int64_t prefill(const int64_t *token_ids, size_t ntoken);
    int64_t step(const int64_t *token_ids, size_t ntoken);
    void resetKVCache();
    void setKVCacheEnabled(bool enabled);

private:
    LlaisysQwen2Meta _meta{};
    const LlaisysQwen2Weights *_weights{nullptr};
    llaisysDeviceType_t _device{LLAISYS_DEVICE_CPU};
    std::vector<int> _device_ids;
    transformer::Decoder _decoder;
};
} // namespace llaisys::models
