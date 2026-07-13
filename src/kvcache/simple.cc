#include "simple.hpp"
#include "../core/context/context.hpp"
#include <cstring>
#include <vector>

namespace llaisys::kvcache::simple {

KVCache::KVCache(usize capacity,
                 usize num_kv_head,
                 usize head_dim,
                 usize vdim,
                 llaisysDataType_t dtype,
                 llaisysDeviceType_t device,
                 int device_id)
    : capacity(capacity), cache_size(0), num_kv_head(num_kv_head),
      head_dim(head_dim), vdim(vdim), dtype(dtype), device(device),
      device_id(device_id),
      keys(Tensor::create(
          {capacity, num_kv_head, head_dim}, dtype, device, device_id)),
      values(Tensor::create(
          {capacity, num_kv_head, vdim}, dtype, device, device_id)) {
    
    // Initialize KV cache to zero
    usize keys_size = capacity * num_kv_head * head_dim * keys->elementSize();
    usize values_size = capacity * num_kv_head * vdim * values->elementSize();
    
    if (device == LLAISYS_DEVICE_CPU) {
        // For CPU, use std::memset directly
        std::memset(static_cast<std::byte *>(keys->data()), 0, keys_size);
        std::memset(static_cast<std::byte *>(values->data()), 0, values_size);
    } else {
        // For other devices (CUDA, etc.), use runtime API
        core::context().setDevice(device, device_id);
        auto *api = core::context().runtime().api();
        
        // Create zero buffers on host and copy to device
        std::vector<char> zero_keys(keys_size, 0);
        std::vector<char> zero_values(values_size, 0);
        
        api->memcpy_sync(keys->data(), zero_keys.data(), keys_size, LLAISYS_MEMCPY_H2D);
        api->memcpy_sync(values->data(), zero_values.data(), values_size, LLAISYS_MEMCPY_H2D);
    }
}

void KVCache::reset() { cache_size = 0; }

void KVCache::insert(const tensor &new_keys,
                     const tensor &new_values,
                     usize n_new) {
    if (cache_size + n_new > capacity)
        throw std::runtime_error("KVCache insert position exceeds capacity.");
    if (new_keys->shape()[0] != n_new || new_keys->shape()[1] != num_kv_head
        || new_keys->shape()[2] != head_dim)
        throw std::runtime_error("New keys tensor shape mismatch.");
    if (new_values->shape()[0] != n_new || new_values->shape()[1] != num_kv_head
        || new_values->shape()[2] != vdim)
        throw std::runtime_error("New values tensor shape mismatch.");
    if (new_keys->deviceType() != device || new_values->deviceType() != device)
        throw std::runtime_error("Device mismatch");
    if (!new_keys->isContiguous() || !new_values->isContiguous())
        throw std::runtime_error("New K/V must be contiguous");
    if (!keys->isContiguous() || !values->isContiguous())
        throw std::runtime_error("Cache storage must be contiguous");

    core::context().setDevice(device, device_id);
    auto *api = core::context().runtime().api();

    // Copy keys
    do {
        auto begin = static_cast<std::byte *>(keys->data())
                   + cache_size * num_kv_head * head_dim * keys->elementSize();
        auto numel = n_new * num_kv_head * head_dim;
        auto size_bytes = numel * new_keys->elementSize();
        
        if (device == LLAISYS_DEVICE_CPU) {
            std::memcpy(begin, static_cast<const std::byte *>(new_keys->data()), size_bytes);
        } else {
            api->memcpy_sync(begin, new_keys->data(), size_bytes, LLAISYS_MEMCPY_D2D);
        }
    } while (false);
    
    // Copy values
    do {
        auto begin = static_cast<std::byte *>(values->data())
                   + cache_size * num_kv_head * vdim * values->elementSize();
        auto numel = n_new * num_kv_head * vdim;
        auto size_bytes = numel * new_values->elementSize();
        
        if (device == LLAISYS_DEVICE_CPU) {
            std::memcpy(begin, static_cast<const std::byte *>(new_values->data()), size_bytes);
        } else {
            api->memcpy_sync(begin, new_values->data(), size_bytes, LLAISYS_MEMCPY_D2D);
        }
    } while (false);

    cache_size += n_new;
}

KVCache::tensor KVCache::getKeysSlice() {
    return keys->slice(0, 0, cache_size);
}

KVCache::tensor KVCache::getValuesSlice() {
    return values->slice(0, 0, cache_size);
}

} // namespace llaisys::kvcache::simple