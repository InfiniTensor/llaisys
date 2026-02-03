#pragma once

#include "llaisys/models/qwen2.h"
#include "llaisys/tensor.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace llaisys::models::transformer {

struct DecoderConfig {
    llaisysDataType_t dtype{};
    size_t nlayer{};
    size_t hs{};
    size_t nh{};
    size_t nkvh{};
    size_t dh{};
    size_t di{};
    size_t maxseq{};
    size_t voc{};
    float epsilon{};
    float theta{};
};

class Decoder {
public:
    Decoder(const DecoderConfig &config,
            const LlaisysQwen2Weights *weights,
            llaisysDeviceType_t device,
            const std::vector<int> &device_ids);
    ~Decoder();

    // Prefill with a full sequence, returns last-step logits.
    bool prefill(const int64_t *token_ids, size_t ntoken, llaisysTensor_t out_last_logits);

    // Decode with only new tokens (append-only), returns last-step logits.
    bool decodeStep(const int64_t *token_ids, size_t ntoken, llaisysTensor_t out_last_logits);

    void resetKVCache();

    void setKVCacheEnabled(bool enabled);

private:
    bool runHidden(const int64_t *token_ids,
                   size_t ntoken,
                   bool append_only,
                   size_t &past_len,
                   size_t &cur_len,
                   llaisysTensor_t &idx,
                   llaisysTensor_t &pos_ids,
                   llaisysTensor_t &hidden);
    void ensureCache();
    void releaseCache();

    DecoderConfig _config{};
    const LlaisysQwen2Weights *_weights{nullptr};
    llaisysDeviceType_t _device{};
    std::vector<int> _device_ids;
    std::vector<llaisysTensor_t> _k_cache;
    std::vector<llaisysTensor_t> _v_cache;
    size_t _past_len{0};
    bool _cache_inited{false};
    bool _kv_cache_enabled{true};
};

} // namespace llaisys::models::transformer
