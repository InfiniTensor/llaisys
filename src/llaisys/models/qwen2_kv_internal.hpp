#pragma once

#include "llaisys/models/qwen2.h"

#include <atomic>
#include <vector>

struct LlaisysQwen2KVBlock {
    LlaisysQwen2KVBlockMeta meta{};
    llaisysDeviceType_t device = LLAISYS_DEVICE_CPU;
    int device_id = 0;
    size_t used_tokens = 0;
    std::vector<llaisysTensor_t> k_layers;
    std::vector<llaisysTensor_t> v_layers;
    std::atomic<int32_t> ref_count{1};
};

struct LlaisysQwen2KVContext {
    llaisysDataType_t dtype = LLAISYS_DTYPE_F32;
    llaisysDeviceType_t device = LLAISYS_DEVICE_CPU;
    int device_id = 0;
    size_t nlayer = 0;
    size_t nh = 0;
    size_t nkvh = 0;
    size_t dh = 0;
    std::vector<LlaisysQwen2KVBlock *> chain;
    std::atomic<int32_t> ref_count{1};
};
