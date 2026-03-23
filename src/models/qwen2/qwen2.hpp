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
    bool prefillPacked(const int64_t *token_ids,
                       size_t ntoken,
                       const int64_t *token_offsets,
                       size_t nseq,
                       int64_t *out_next_tokens);
    bool stepPacked(const int64_t *token_ids,
                    size_t ntoken,
                    const int64_t *token_offsets,
                    size_t nseq,
                    int64_t *out_next_tokens);
    int64_t prefillSampling(const int64_t *token_ids, size_t ntoken, const LlaisysSamplingParams *params);
    int64_t stepSampling(const int64_t *token_ids, size_t ntoken, const LlaisysSamplingParams *params);
    void resetKVCache();
    void setKVCacheEnabled(bool enabled);
    void setTensorParallel(llaisysComm_t comm, llaisysStream_t stream, int tp_size);
    void setKVContext(void *ctx, size_t past_len_tokens = 0);
    void *getKVContext() const;
    int exportKVContext(void *ctx, size_t block_tokens);

private:
    void clearPackedState();

    LlaisysQwen2Meta _meta{};
    const LlaisysQwen2Weights *_weights{nullptr};
    llaisysDeviceType_t _device{LLAISYS_DEVICE_CPU};
    std::vector<int> _device_ids;
    transformer::Decoder _decoder;
    void *_kv_ctx{nullptr};
    std::vector<LlaisysQwen2KVContext *> _packed_kv_contexts;
    std::vector<std::vector<int64_t>> _packed_prompts;
};
} // namespace llaisys::models
