#pragma once
#include "llaisys/comm.h"

#include "../utils.hpp"

namespace llaisys::device {
const LlaisysCommAPI *getCommAPI(llaisysCommBackend_t backend);
int commGenerateUniqueId(llaisysCommBackend_t backend, void *id_out, size_t *id_size);

const LlaisysCommAPI *getUnsupportedCommAPI();

#ifdef ENABLE_NVIDIA_API
namespace nccl {
const LlaisysCommAPI *getCommAPI();
int commGenerateUniqueId(void *id_out, size_t *id_size);
}
#endif

#ifdef ENABLE_ILUVATAR_API
namespace ixccl {
const LlaisysCommAPI *getCommAPI();
}
#endif

} // namespace llaisys::device
