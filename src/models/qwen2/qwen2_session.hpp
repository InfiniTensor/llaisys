#include <memory>
#include <vector>
#include "../core/tensor.hpp"
#include "qwen2_kvcache.hpp"

namespace llaisys {

struct Qwen2Config;

class Qwen2Session {
public:
    Qwen2Session(const Qwen2Config &config, DeviceType device);
    ~Qwen2Session() = default;

    Qwen2KVCache &kv_cache() { return _kv_cache; }
    const Qwen2KVCache &kv_cache() const { return _kv_cache; }

    void reset() { _kv_cache.reset(); }

private:
    Qwen2KVCache _kv_cache;
};

} // namespace llaisys
