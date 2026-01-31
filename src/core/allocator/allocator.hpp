#pragma once

#include "llaisys/runtime.h"

#include "../storage/storage.hpp"

namespace llaisys::core {
class MemoryAllocator {
protected:
    const LlaisysRuntimeAPI *_api;
    MemoryAllocator(const LlaisysRuntimeAPI *runtime_api) : _api(runtime_api){};

public:
    virtual ~MemoryAllocator() = default;
    virtual std::byte *allocate(size_t size) = 0;
    virtual void release(std::byte *memory) = 0;
};

} // namespace llaisys::core
