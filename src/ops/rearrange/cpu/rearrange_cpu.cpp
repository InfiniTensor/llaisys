#include "rearrange_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>
#include <numeric>
#include <vector>

namespace {
void gather_contiguous(const std::byte *in,
                       std::byte *buf,
                       const std::vector<size_t> &shape,
                       const std::vector<ptrdiff_t> &in_strides,
                       size_t elem_size,
                       size_t dim,
                       ptrdiff_t in_offset,
                       size_t &write_idx) {
    if (dim == shape.size()) {
        std::memcpy(buf + write_idx * elem_size, in + in_offset * elem_size, elem_size);
        ++write_idx;
        return;
    }
    for (size_t i = 0; i < shape[dim]; ++i) {
        gather_contiguous(in, buf, shape, in_strides, elem_size, dim + 1,
                          in_offset + static_cast<ptrdiff_t>(i) * in_strides[dim], write_idx);
    }
}

void scatter_from_contiguous(const std::byte *buf,
                             std::byte *out,
                             const std::vector<size_t> &shape,
                             const std::vector<ptrdiff_t> &out_strides,
                             size_t elem_size,
                             size_t dim,
                             ptrdiff_t out_offset,
                             size_t &read_idx) {
    if (dim == shape.size()) {
        std::memcpy(out + out_offset * elem_size, buf + read_idx * elem_size, elem_size);
        ++read_idx;
        return;
    }
    for (size_t i = 0; i < shape[dim]; ++i) {
        scatter_from_contiguous(buf, out, shape, out_strides, elem_size, dim + 1,
                                out_offset + static_cast<ptrdiff_t>(i) * out_strides[dim], read_idx);
    }
}
} // namespace

namespace llaisys::ops::cpu {
void rearrange(std::byte *out,
               const std::byte *in,
               const std::vector<size_t> &shape,
               const std::vector<ptrdiff_t> &out_strides,
               const std::vector<ptrdiff_t> &in_strides,
               llaisysDataType_t dtype) {
    const size_t elem_size = llaisys::utils::dsize(dtype);

    const size_t total = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
    std::vector<std::byte> buffer(total * elem_size);

    size_t write_idx = 0;
    gather_contiguous(in, buffer.data(), shape, in_strides, elem_size, /*dim=*/0, /*in_offset=*/0, write_idx);

    size_t read_idx = 0;
    scatter_from_contiguous(buffer.data(), out, shape, out_strides, elem_size, /*dim=*/0, /*out_offset=*/0, read_idx);
}
} // namespace llaisys::ops::cpu
