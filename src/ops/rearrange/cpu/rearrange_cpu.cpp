#include "rearrange_cpu.hpp"

#include <cstring>

namespace {
void rearrange_recursive(std::byte *out,
                         const std::byte *in,
                         const std::vector<size_t> &shape,
                         const std::vector<ptrdiff_t> &out_strides,
                         const std::vector<ptrdiff_t> &in_strides,
                         size_t elem_size,
                         size_t dim,
                         ptrdiff_t out_off,
                         ptrdiff_t in_off) {
    if (dim == shape.size()) {
        std::memcpy(out + out_off * elem_size, in + in_off * elem_size, elem_size);
        return;
    }

    const size_t len = shape[dim];
    const ptrdiff_t os = out_strides[dim];
    const ptrdiff_t is = in_strides[dim];

    for (size_t i = 0; i < len; ++i) {
        rearrange_recursive(out,
                            in,
                            shape,
                            out_strides,
                            in_strides,
                            elem_size,
                            dim + 1,
                            out_off + static_cast<ptrdiff_t>(i) * os,
                            in_off + static_cast<ptrdiff_t>(i) * is);
    }
}
} // namespace

namespace llaisys::ops::cpu {
void rearrange(std::byte *out,
               const std::byte *in,
               const std::vector<size_t> &shape,
               const std::vector<ptrdiff_t> &out_strides,
               const std::vector<ptrdiff_t> &in_strides,
               size_t elem_size) {
    rearrange_recursive(out, in, shape, out_strides, in_strides, elem_size, 0, 0, 0);
}
} // namespace llaisys::ops::cpu
