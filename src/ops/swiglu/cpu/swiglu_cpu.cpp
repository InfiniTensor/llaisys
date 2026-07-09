#include "swiglu_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

template <typename T>
void swiglu_(T *out, const T *gate, const T *up, size_t seqlen, size_t intermediate_size) {
	const size_t max_idx = seqlen * intermediate_size;
    if constexpr (std::is_same_v<T, llaisys::fp16_t> || std::is_same_v<T, llaisys::bf16_t>) {
		for (size_t i = 0; i < max_idx; i++) {
			float gate_f = llaisys::utils::cast<float>(gate[i]);
			float up_f = llaisys::utils::cast<float>(up[i]);
			float sigmoid = 1.0f / (1.0f + std::exp(-gate_f));
			out[i] = llaisys::utils::cast<T>(up_f * sigmoid * gate_f);
		}
    } else {
		for (size_t i = 0; i < max_idx; i++) {
			out[i] = up[i] * (1.0f / (1.0f + std::exp(-gate[i]))) * gate[i];
		}	
    }
}

namespace llaisys::ops::cpu {
void swiglu(std::byte *out, std::byte *gate, std::byte *up, llaisysDataType_t dtype, size_t seqlen, size_t intermediate_size) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return swiglu_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(gate), reinterpret_cast<const float *>(up), seqlen, intermediate_size);
    case LLAISYS_DTYPE_F16:
        return swiglu_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(gate), reinterpret_cast<const llaisys::fp16_t *>(up), seqlen, intermediate_size);
    case LLAISYS_DTYPE_BF16:
        return swiglu_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(gate), reinterpret_cast<const llaisys::bf16_t *>(up), seqlen, intermediate_size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
