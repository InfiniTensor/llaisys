#include "rearrange_cpu.hpp"
#include "../../../utils.hpp"

#include <cstring> // std::memcpy
#include <numeric>
#include <vector>

namespace {

inline bool is_contiguous_like(const std::vector<size_t> &shape,
                               const std::vector<ptrdiff_t> &strides) {
    // row-major
    size_t n = shape.size();
    size_t expected = 1;
    for (size_t i = n; i-- > 0;) {
        if (shape[i] <= 1) {
            continue;
        }
        if (static_cast<size_t>(strides[i]) != expected) {
            return false;
        }
        expected *= shape[i];
    }
    return true;
}

} // anonymous namespace

namespace llaisys::ops::cpu {

void rearrange(std::byte *dst,
               const std::byte *src,
               llaisysDataType_t dtype,
               const std::vector<size_t> &shape,
               const std::vector<ptrdiff_t> &dst_strides,
               const std::vector<ptrdiff_t> &src_strides) {
    const size_t ndim = shape.size();
    const size_t esize = llaisys::utils::dsize(dtype);

    // numel
    size_t numel = 1;
    for (size_t s : shape) {
        numel *= s;
    }

    if (numel == 0) {
        return;
    }


    if (is_contiguous_like(shape, dst_strides) && is_contiguous_like(shape, src_strides)) {
        std::memcpy(dst, src, numel * esize);
        return;
    }


    std::vector<size_t> idx(ndim, 0);

    while (true) {
        
        ptrdiff_t src_off_elems = 0;
        ptrdiff_t dst_off_elems = 0;
        for (size_t i = 0; i < ndim; ++i) {
            src_off_elems += static_cast<ptrdiff_t>(idx[i]) * src_strides[i];
            dst_off_elems += static_cast<ptrdiff_t>(idx[i]) * dst_strides[i];
        }

        std::byte *dptr = dst + static_cast<ptrdiff_t>(esize) * dst_off_elems;
        const std::byte *sptr = src + static_cast<ptrdiff_t>(esize) * src_off_elems;
        std::memcpy(dptr, sptr, esize);

        // 
        for (size_t k = ndim; k-- > 0;) {
            if (++idx[k] < shape[k]) {
                break;
            }
            if (k == 0) {
                
                return;
            }
            idx[k] = 0;
        }
    }
}

} // namespace llaisys::ops::cpu
