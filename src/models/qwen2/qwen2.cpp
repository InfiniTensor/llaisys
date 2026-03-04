#include "qwen2.hpp"

#include "llaisys/ops.h"

#include "../../utils.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace llaisys::models {
Qwen2::Qwen2(const LlaisysQwen2Meta &meta,
             const LlaisysQwen2Weights &weights,
             llaisysDeviceType_t device,
             const std::vector<int> &device_ids)
    : _meta(meta),
      _weights(&weights),
      _device(device),
      _device_ids(device_ids),
      _decoder(transformer::DecoderConfig{
                   meta.dtype,
                   meta.nlayer,
                   meta.hs,
                   meta.nh,
                   meta.nkvh,
                   meta.dh,
                   meta.di,
                   meta.maxseq,
                   meta.voc,
                   meta.epsilon,
                   meta.theta},
               &weights,
               device,
               device_ids) {}

Qwen2::~Qwen2() {
}

void Qwen2::resetKVCache() {
    _decoder.resetKVCache();
}

void Qwen2::setKVCacheEnabled(bool enabled) {
    _decoder.setKVCacheEnabled(enabled);
}

// 执行 Qwen2 模型的推理并返回下一个 token（使用贪心策略）
static int64_t argmax_from_logits(llaisysTensor_t logits,
                                  llaisysDataType_t dtype,
                                  llaisysDeviceType_t device,
                                  int device_id) {
    int64_t next_token = -1;
    size_t one_shape[1] = {1};
    llaisysTensor_t max_idx = tensorCreate(one_shape, 1, LLAISYS_DTYPE_I64, device, device_id);
    llaisysTensor_t max_val = tensorCreate(one_shape, 1, dtype, device, device_id);
    if (!max_idx || !max_val) {
        if (max_idx) tensorDestroy(max_idx);
        if (max_val) tensorDestroy(max_val);
        return -1;
    }
    ::llaisysArgmax(max_idx, max_val, logits);
    if (tensorGetDeviceType(max_idx) == LLAISYS_DEVICE_CPU) {
        next_token = *reinterpret_cast<int64_t *>(tensorGetData(max_idx));
    }
    tensorDestroy(max_idx);
    tensorDestroy(max_val);
    return next_token;
}

int64_t Qwen2::infer(const int64_t *token_ids, size_t ntoken) {
    return prefill(token_ids, ntoken);
}

int64_t Qwen2::prefill(const int64_t *token_ids, size_t ntoken) {
    if (!token_ids || ntoken == 0) return -1;

    const int device_id = _device_ids.empty() ? 0 : _device_ids[0];
    size_t logits_shape[2] = {1, _meta.voc};
    llaisysTensor_t logits = tensorCreate(logits_shape, 2, _meta.dtype, _device, device_id);
    if (!logits) return -1;
    if (!_decoder.prefill(token_ids, ntoken, logits)) {
        tensorDestroy(logits);
        return -1;
    }

    int64_t next_token = argmax_from_logits(logits, _meta.dtype, _device, device_id);
    tensorDestroy(logits);

    return next_token;
}

int64_t Qwen2::step(const int64_t *token_ids, size_t ntoken) {
    if (!token_ids || ntoken == 0) return -1;

    const int device_id = _device_ids.empty() ? 0 : _device_ids[0];
    size_t logits_shape[2] = {1, _meta.voc};
    llaisysTensor_t logits = tensorCreate(logits_shape, 2, _meta.dtype, _device, device_id);
    if (!logits) return -1;
    if (!_decoder.decodeStep(token_ids, ntoken, logits)) {
        tensorDestroy(logits);
        return -1;
    }

    int64_t next_token = argmax_from_logits(logits, _meta.dtype, _device, device_id);
    tensorDestroy(logits);
    return next_token;
}
} // namespace llaisys::models