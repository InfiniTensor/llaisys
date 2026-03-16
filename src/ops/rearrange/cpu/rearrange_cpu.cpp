#include "rearrange_cpu.hpp"

#include <cstring>

namespace llaisys::ops::cpu {
void rearrange(std::byte *out, const std::byte *in,
               const std::vector<size_t> &shape,
               const std::vector<ptrdiff_t> &out_strides,
               const std::vector<ptrdiff_t> &in_strides,
               size_t esize, size_t numel) {
    size_t ndim = shape.size();
    std::vector<size_t> idx(ndim, 0);

    for (size_t i = 0; i < numel; ++i) {
        ptrdiff_t src_off = 0, dst_off = 0;
        for (size_t d = 0; d < ndim; ++d) {
            src_off += idx[d] * in_strides[d];
            dst_off += idx[d] * out_strides[d];
        }
        std::memcpy(out + dst_off * esize, in + src_off * esize, esize);

        for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
            if (++idx[d] < shape[d]) break;
            idx[d] = 0;
        }
    }
}
} // namespace llaisys::ops::cpu
