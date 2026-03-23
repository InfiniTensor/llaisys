#include "llaisys/comm.h"
#include "../device/comm_api.hpp"

__C const LlaisysCommAPI *llaisysGetCommAPI(llaisysCommBackend_t backend) {
    return llaisys::device::getCommAPI(backend);
}

__C int llaisysCommGenerateUniqueId(llaisysCommBackend_t backend, void *id_out, size_t *id_size) {
    return llaisys::device::commGenerateUniqueId(backend, id_out, id_size);
}
