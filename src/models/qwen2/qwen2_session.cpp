#include "qwen2_session.hpp"
#include "qwen2_model.hpp" // For Qwen2Config definition

namespace llaisys {

Qwen2Session::Qwen2Session(const Qwen2Config &config, DeviceType device)
    : _kv_cache(config.nlayers, config.maxseq, config.nkvhead, config.head_size, device) {
}

} // namespace llaisys
