#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <limits>

template <typename T>
void argmax_(int64_t *mi, T *mv, const T *v, size_t numel) {
    mi[0]=0;
    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        mv[0] = llaisys::utils::cast<T>(llaisys::utils::cast<float>(v[0]));

        for (size_t i = 1; i < numel; i++) {
            if(llaisys::utils::cast<float>(mv[0]) < llaisys::utils::cast<float>(v[i])){
                    mi[0] = static_cast<int64_t>(i);
                    mv[0] = v[i];
            }
        }

    }else{
        mv[0] = v[0];
        for (size_t i = 1; i < numel; i++) {
            if(mv[0] < v[i]){
                    mi[0] = static_cast<int64_t>(i);
                    mv[0] = v[i];
            }
        }
    }
}
namespace llaisys::ops::cpu {
void argmax(std::byte *mi, std::byte *mv, const std::byte *v, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_(reinterpret_cast<int64_t *>(mi), reinterpret_cast<float *>(mv), reinterpret_cast<const float *>(v), numel);
    case LLAISYS_DTYPE_BF16:
        return argmax_(reinterpret_cast<int64_t *>(mi), reinterpret_cast<llaisys::bf16_t *>(mv),
                    reinterpret_cast<const llaisys::bf16_t *>(v), numel);
    case LLAISYS_DTYPE_F16:
        return argmax_(reinterpret_cast<int64_t *>(mi), reinterpret_cast<llaisys::fp16_t *>(mv),
                    reinterpret_cast<const llaisys::fp16_t *>(v), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
