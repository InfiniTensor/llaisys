#pragma once

#include "allocator.hpp"

namespace llaisys::core::allocators {
class NaiveAllocator : public MemoryAllocator {
public:
    NaiveAllocator(const LlaisysRuntimeAPI *runtime_api);
    ~NaiveAllocator() = default;
    std::byte *allocate(size_t size) override;
    void release(std::byte *memory) override;
};
} // namespace llaisys::core::allocators