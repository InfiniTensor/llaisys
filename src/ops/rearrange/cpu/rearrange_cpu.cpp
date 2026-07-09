#include "rearrange_cpu.hpp"
#include <cstring>

namespace llaisys::ops::cpu {

void rearrange_recursive(std::byte *out, const std::byte *in, const std::vector<size_t> &shape,
                        const std::vector<ptrdiff_t> &in_strides, size_t elem_size,
                        size_t dim, size_t &out_offset, size_t in_offset) {
    if (dim == shape.size()) {
        std::memcpy(out + out_offset, in + in_offset, elem_size);
        out_offset += elem_size;
        return;
    }
    
    for (size_t i = 0; i < shape[dim]; i++) {
        rearrange_recursive(out, in, shape, in_strides, elem_size, dim + 1, out_offset,
                           in_offset + i * in_strides[dim] * elem_size);
    }
}

void rearrange(std::byte *out, const std::byte *in, const std::vector<size_t> &shape,
               const std::vector<ptrdiff_t> &in_strides, size_t elem_size) {
    size_t out_offset = 0;
    rearrange_recursive(out, in, shape, in_strides, elem_size, 0, out_offset, 0);
}

}
