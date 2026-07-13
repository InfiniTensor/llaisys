#pragma once

#include "../tensor/tensor.hpp"
#include "llaisys.h"
#include <cstddef>

namespace llaisys::kvcache::simple {

/**
 * @brief A simple per-layer KV cache implementation for transformer models.
 */
struct KVCache {
    using usize = size_t;
    using tensor = tensor_t;

    usize capacity;
    usize cache_size;

    usize num_kv_head;
    usize head_dim;
    usize vdim;
    llaisysDataType_t dtype;
    llaisysDeviceType_t device;
    int device_id;

    tensor keys;
    tensor values;

    KVCache(usize capacity,
            usize num_kv_head,
            usize head_dim,
            usize vdim,
            llaisysDataType_t dtype,
            llaisysDeviceType_t device,
            int device_id);
    ~KVCache() = default;

    /**
     * @brief Reset the KV cache to empty state.
     */
    void reset();

    /**
     * @brief Insert new keys and values to the cache at the given position.
     * @note This function might overwrite existing entries.
     *
     * @param new_keys The new keys to insert. Shape: [n_new, num_kv_head, head_dim]
     * @param new_values The new values to insert. Shape: [n_new, num_kv_head, vdim]
     * @param n_new Number of new key-value pairs to insert.
     * @param insert_pos Position to insert the new key-value pairs.
     */
    void insert(const tensor &new_keys, const tensor &new_values, usize n_new);

    /**
     * @brief get a slice of keys tensor up to the current cache size.
     */
    tensor getKeysSlice();
    /**
     * @brief get a slice of values tensor up to the current cache size.
     */
    tensor getValuesSlice();

    usize getCacheSize() const { return cache_size; }
};

} // namespace llaisys::kvcache::simple