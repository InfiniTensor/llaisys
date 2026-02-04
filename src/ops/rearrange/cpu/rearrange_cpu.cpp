#include "rearrange_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>

namespace {

static inline size_t numel_(const size_t *shape, size_t ndim) {
    size_t n = 1;
    for (size_t i = 0; i < ndim; i++) n *= shape[i];
    return n;
}

// Compute the offset in elements for a given linear index under `shape` and `strides`.
static inline size_t offset_from_linear_(size_t linear,
                                        const size_t *shape,
                                        const ptrdiff_t *strides,
                                        size_t ndim) {
    // Unravel `linear` into indices, then dot with strides.
    //
    // strides are in elements (not bytes) and may be non-contiguous. We assume
    // non-negative strides for now (the framework doesn't support negative strides yet).
    ptrdiff_t off = 0;
    for (size_t d = 0; d < ndim; d++) {
        const size_t dim = ndim - 1 - d; // last dim first
        const size_t size_d = shape[dim];
        const size_t idx_d = linear % size_d;
        linear /= size_d;
        off += static_cast<ptrdiff_t>(idx_d) * strides[dim];
    }
    CHECK_ARGUMENT(off >= 0, "rearrange_cpu: negative offset (negative strides not supported)");
    return static_cast<size_t>(off);
}

} // namespace

namespace llaisys::ops::cpu {

void rearrange(std::byte *out,
               const std::byte *in,
               llaisysDataType_t dtype,
               const size_t *shape,
               const ptrdiff_t *out_strides,
               const ptrdiff_t *in_strides,
               size_t ndim) {
    CHECK_ARGUMENT(out != nullptr && in != nullptr, "rearrange_cpu: null data ptr");
    CHECK_ARGUMENT(shape != nullptr && out_strides != nullptr && in_strides != nullptr, "rearrange_cpu: null meta ptr");
    CHECK_ARGUMENT(ndim > 0, "rearrange_cpu: ndim must be > 0");

    const size_t esize = llaisys::utils::dsize(dtype);
    const size_t n = numel_(shape, ndim);

    for (size_t linear = 0; linear < n; linear++) {
        const size_t in_off = offset_from_linear_(linear, shape, in_strides, ndim);
        const size_t out_off = offset_from_linear_(linear, shape, out_strides, ndim);
        std::memcpy(out + out_off * esize, in + in_off * esize, esize);
    }
}

} // namespace llaisys::ops::cpu
