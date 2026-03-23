#pragma once

#include "llaisys/models/qwen2.h"
#include "llaisys/comm.h"
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
    // Prefill packed independent sequences, outputs one logits row per sequence.
    bool prefillPacked(const int64_t *token_ids,
                       size_t ntoken,
                       const int64_t *token_offsets,
                       size_t nseq,
                       llaisysTensor_t out_last_logits);

    // Decode with only new tokens (append-only), returns last-step logits.
    bool decodeStep(const int64_t *token_ids, size_t ntoken, llaisysTensor_t out_last_logits);
    // Decode one token per sequence in a packed batch with per-sequence KV contexts.
    bool decodePacked(const int64_t *token_ids,
                      size_t nseq,
                      const std::vector<LlaisysQwen2KVContext *> &contexts,
                      llaisysTensor_t out_last_logits,
                      size_t block_tokens_hint);

    void resetKVCache();

    void setKVCacheEnabled(bool enabled);
    void bindExternalKVContext(void *ctx, size_t past_len_tokens);
    void clearExternalKVContext();
    bool hasExternalKVContext() const;
    int exportKVContext(void *ctx, size_t block_tokens);

    void setTensorParallel(llaisysComm_t comm, llaisysStream_t stream, int tp_size);

private:
    bool recoverExternalCache();
    bool runHidden(const int64_t *token_ids,
                   size_t ntoken,
                   bool append_only,
                   const int64_t *segment_offsets,
                   size_t nseg,
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
    void *_external_kv_ctx{nullptr};
    size_t _external_past_len{0};
    bool _external_cache_ready{false};
    llaisysComm_t _comm{nullptr};
    llaisysStream_t _comm_stream{nullptr};
    int _tp_size{1};
};

} // namespace llaisys::models::transformer
