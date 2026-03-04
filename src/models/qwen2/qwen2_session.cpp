#include "qwen2_session.hpp"
#include "qwen2_model.hpp"

namespace llaisys::models::qwen2 {

Qwen2Session::Qwen2Session(const Qwen2Config &config, llaisysDeviceType_t device) {
    _kv_cache.reserve(config.nlayers, config.maxseq, config.nkvhead, config.head_size, LLAISYS_DTYPE_BF16, device, 0);
}

} // namespace llaisys::models::qwen2
