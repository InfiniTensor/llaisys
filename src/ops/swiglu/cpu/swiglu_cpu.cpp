#include "swiglu_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

namespace llaisys::ops::cpu {

template <typename T>
// 扁平化处理 (numel)：
// SwiGLU 是一个逐元素操作。这意味着第 i 个位置的输出只取决于第 i 个位置的输入，与周围的数据无关。
// 因此，无论输入张量的形状是[Batch,Seq,Dim]还是[Seq, Dim]，在内存连续的情况下，
// 都可以将其视为一个长度为numel的一维大数组来处理。这简化了循环逻辑。

void swiglu_kernel(T *out, const T *gate, const T *up, size_t numel) {
    // SwiGLU 是逐元素操作，直接并行化整个数据量
    #pragma omp parallel for
    for (size_t i = 0; i < numel; ++i) {
        float g = llaisys::utils::cast<float>(gate[i]);
        float u = llaisys::utils::cast<float>(up[i]);
        
        // 公式: out = up * (gate / (1 + exp(-gate)))
        // 即 out = up * SiLU(gate)
        float silu = g / (1.0f + std::exp(-g));
        
        out[i] = llaisys::utils::cast<T>(u * silu);
    }
}

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t dtype, size_t numel) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        swiglu_kernel<float>((float *)out, (const float *)gate, (const float *)up, numel);
        break;
    case LLAISYS_DTYPE_F16:
        swiglu_kernel<llaisys::fp16_t>((llaisys::fp16_t *)out, (const llaisys::fp16_t *)gate, (const llaisys::fp16_t *)up, numel);
        break;
    case LLAISYS_DTYPE_BF16:
        swiglu_kernel<llaisys::bf16_t>((llaisys::bf16_t *)out, (const llaisys::bf16_t *)gate, (const llaisys::bf16_t *)up, numel);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu