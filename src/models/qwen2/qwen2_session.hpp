#pragma once

#include <memory>
#include <vector>
#include "../../tensor/tensor.hpp"
#include "../../utils.hpp"
#include "qwen2_kvcache.hpp"

namespace llaisys::models::qwen2 {

struct Qwen2Config;

class Qwen2Session {
public:
    Qwen2Session(const Qwen2Config &config, llaisysDeviceType_t device);
    ~Qwen2Session() = default;

    Qwen2KVCache &kv_cache() { return _kv_cache; }
    const Qwen2KVCache &kv_cache() const { return _kv_cache; }

    void reset() { _kv_cache.reset(); }

private:
    Qwen2KVCache _kv_cache;
};

} // namespace llaisys::models::qwen2
