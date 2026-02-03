#include "qwen2_kvcache.hpp"

namespace llaisys::models::qwen2 {

void Qwen2KVCache::reset() {
    _seq_len = 0;
}

void Qwen2KVCache::reserve(
    size_t nlayer,
    size_t maxseq,
    size_t nkvh,
    size_t dh,
    llaisysDataType_t dtype,
    llaisysDeviceType_t device,
    int device_id) {
    if (_k.size() == nlayer && _v.size() == nlayer) {
        return;
    }

    _k.clear();
    _v.clear();
    _k.reserve(nlayer);
    _v.reserve(nlayer);

    for (size_t i = 0; i < nlayer; ++i) {
        std::vector<size_t> shape = {maxseq, nkvh, dh};
        _k.push_back(Tensor::create(shape, dtype, device, device_id));
        _v.push_back(Tensor::create(shape, dtype, device, device_id));
    }
}

void Qwen2KVCache::advance(size_t n) {
    _seq_len += n;
}

tensor_t Qwen2KVCache::k(size_t layer) const {
    CHECK_ARGUMENT(layer < _k.size(), "KV cache layer out of range");
    return _k[layer];
}

tensor_t Qwen2KVCache::v(size_t layer) const {
    CHECK_ARGUMENT(layer < _v.size(), "KV cache layer out of range");
    return _v[layer];
}

} // namespace llaisys::models::qwen2
