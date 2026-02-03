#pragma once

#include "../../tensor/tensor.hpp"
#include "../../utils.hpp"

#include <vector>

namespace llaisys::models::qwen2 {
class Qwen2KVCache {
public:
    Qwen2KVCache() = default;

    void reset();
    void reserve(
        size_t nlayer,
        size_t maxseq,
        size_t nkvh,
        size_t dh,
        llaisysDataType_t dtype,
        llaisysDeviceType_t device,
        int device_id);

    size_t seq_len() const { return _seq_len; }
    void advance(size_t n);

    tensor_t k(size_t layer) const;
    tensor_t v(size_t layer) const;

private:
    size_t _seq_len = 0;
    std::vector<tensor_t> _k;
    std::vector<tensor_t> _v;
};
} // namespace llaisys::models::qwen2
