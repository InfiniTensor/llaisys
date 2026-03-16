// 系统 SIMD 头文件须在项目头文件之前包含，避免项目宏 __C 与系统头文件中的 __C 冲突
#ifdef __AVX2__
#include <immintrin.h>
#endif
#include <cstddef>

#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#ifdef ENABLE_NVIDIA_API
#include "llaisys/ops_nvidia.h"
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

// 线性层（矩阵乘法）核心实现，计算公式为 Y = X W^T + b。
// 参数说明：out 形状为 (B, M)，in 形状为 (B, K)，weight 形状为 (M, K)，bias 形状为 (M,) 或者是空指针。
// 注意 weight 的形状是 (M, K) 而不是 (K, M)，这表明权重在内存中是以转置后的形态连续存储的。
// OpenMP：外层 B 维并行，多核同时计算多行输出。
template <typename T>
void linear_impl(T *out, const T *in, const T *weight, const T *bias, size_t B, size_t M, size_t K) {
    // 外层循环：遍历输入张量的批次大小或序列长度维度 B。
    // MSVC OpenMP 要求 index 为 signed integral type，故用 ptrdiff_t。
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(B); i++) {
        // 中层循环：遍历输出特征维度 M。
        for (size_t j = 0; j < M; j++) {
            
            // 编译期条件分支：if constexpr 是 C++17 特性，用于在编译阶段静态计算表达式。
            // std::is_same_v 用于类型萃取，判断当前模板类型 T 是否为 16 位浮点数（bf16_t 或 fp16_t）。
            // 这种写法确保了在程序运行时，这里没有任何 if/else 的条件跳转指令开销，极大优化了指令执行效率。
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                
                // 声明一个 32 位单精度浮点数 sum_f 作为累加器。
                // 16 位浮点数的尾数位数较少，如果在长循环的矩阵点积中直接用 16 位类型进行累加，
                // 会产生严重的舍入误差，甚至可能导致数值溢出。使用 float 作为高精度中间变量是深度学习推理的通用做法。
                float sum_f = 0; 
                
                // 内层循环：执行向量点积计算。遍历特征维度 K。
                for (size_t k = 0; k < K; k++)
                    // 内存访问模式解析：
                    // in[i * K + k] 是按行主序访问输入的第 i 行第 k 列。
                    // weight[j * K + k] 是按行主序访问权重的第 j 行第 k 列。
                    // 由于计算公式是 X * W^T，正常应该访问权重的第 k 行第 j 列，但因为我们的 weight 传入时就是 (M, K) 形状，
                    // 这使得我们在内层循环 k 递增时，in 和 weight 的内存地址都是线性且连续递增的。
                    // 这种连续的内存访问模式能够最大化 CPU L1/L2 缓存的命中率（Cache Prefetching），是性能优化的关键。
                    sum_f += llaisys::utils::cast<float>(in[i * K + k]) * llaisys::utils::cast<float>(weight[j * K + k]);
                
                // 如果传入了偏置项指针，将其对应元素转换为 float 后加到累加器上。
                if (bias) sum_f += llaisys::utils::cast<float>(bias[j]);
                
                // 将 32 位的高精度累加结果强制向下转换为原始的 16 位类型 T，并写入输出张量。
                out[i * M + j] = llaisys::utils::cast<T>(sum_f);
                
            } else {
                // 针对 FP32 等全精度类型的处理逻辑。
                // 因为本身精度足够，直接使用类型 T 初始化累加器，避免了频繁的类型转换开销。
                T sum = T(0);
                for (size_t k = 0; k < K; k++)
                    sum += in[i * K + k] * weight[j * K + k];
                if (bias) sum += bias[j];
                out[i * M + j] = sum;
            }
        }
    }
}

#ifdef __AVX2__
static void linear_f32_avx2(float *out, const float *in, const float *weight, const float *bias,
                            size_t B, size_t M, size_t K);
#endif

// 运行时类型分发函数：将无类型的底层字节指针转换为对应类型的指针，并调用模板函数。
void linear_cpu(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
                llaisysDataType_t dtype, size_t B, size_t M, size_t K) {
    // switch 语句根据张量元数据中的 dtype 字段进行运行时路由。
    switch (dtype) {
    case LLAISYS_DTYPE_F32: {
        float *out_f = reinterpret_cast<float *>(out);
        const float *in_f = reinterpret_cast<const float *>(in);
        const float *weight_f = reinterpret_cast<const float *>(weight);
        const float *bias_f = bias ? reinterpret_cast<const float *>(bias) : nullptr;
#ifdef __AVX2__
        linear_f32_avx2(out_f, in_f, weight_f, bias_f, B, M, K);
#else
        linear_impl(out_f, in_f, weight_f, bias_f, B, M, K);
#endif
        return;
    }
    case LLAISYS_DTYPE_F16:
        linear_impl(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                    reinterpret_cast<const llaisys::fp16_t *>(weight), bias ? reinterpret_cast<const llaisys::fp16_t *>(bias) : nullptr, B, M, K);
        return;
    case LLAISYS_DTYPE_BF16:
        linear_impl(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                    reinterpret_cast<const llaisys::bf16_t *>(weight), bias ? reinterpret_cast<const llaisys::bf16_t *>(bias) : nullptr, B, M, K);
        return;
    default:
        // 遇到不支持的数值类型，直接抛出异常中断执行。
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

#ifdef __AVX2__
// FP32 专用：内层 K 维用 AVX2 一次处理 8 个 float，提升缓存与 SIMD 利用率。
static inline float hsum_avx(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    lo = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, lo);
    lo = _mm_add_ss(lo, shuf);
    return _mm_cvtss_f32(lo);
}
static void linear_f32_avx2(float *out, const float *in, const float *weight, const float *bias,
                            size_t B, size_t M, size_t K) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(B); i++) {
        const float *in_row = in + i * K;
        for (size_t j = 0; j < M; j++) {
            const float *w_row = weight + j * K;
            __m256 sum8 = _mm256_setzero_ps();
            size_t k = 0;
            for (; k + 8 <= K; k += 8) {
                __m256 a = _mm256_loadu_ps(in_row + k);
                __m256 b = _mm256_loadu_ps(w_row + k);
                sum8 = _mm256_fmadd_ps(a, b, sum8);
            }
            float sum = hsum_avx(sum8);
            for (; k < K; k++)
                sum += in_row[k] * w_row[k];
            if (bias) sum += bias[j];
            out[i * M + j] = sum;
        }
    }
}
#endif

} // namespace

namespace llaisys::ops {

// Linear 算子的对外 API 入口，负责严格的参数合法性校验和硬件设备调度。
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    // 设备一致性校验：确保输入、权重、输出张量所在的物理设备（如均在 CPU 内存中）完全一致，防止跨设备非同步访问导致的段错误。
    CHECK_SAME_DEVICE(out, in, weight);
    if (bias) CHECK_SAME_DEVICE(out, bias);
    
    // 数据类型一致性校验：深度学习推理中，参与同一次矩阵运算的张量精度通常必须对齐。
    ASSERT(out->dtype() == in->dtype() && out->dtype() == weight->dtype(), "linear: out, in, weight must have same dtype");
    if (bias) ASSERT(out->dtype() == bias->dtype(), "linear: bias dtype must match");
    
    // 内存连续性校验：极度关键。
    // 上方的 linear_impl 实现强依赖于 1D 数组的线性索引（如 i * K + k）来模拟 2D 矩阵访问。
    // 如果张量经过了 view 或 permute 等操作导致物理内存不连续，直接将指针传入底层计算会导致索引越界或读取到错误数据。
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "linear: out, in, weight must be contiguous");
    if (bias) ASSERT(bias->isContiguous() && bias->ndim() == 1, "linear: bias must be 1D contiguous");
    
    // 维度数量校验：确保矩阵乘法的基础要求，输入输出均为 2D 矩阵（通常为 [Batch/SeqLen, Features]）。
    ASSERT(out->ndim() == 2 && in->ndim() == 2 && weight->ndim() == 2, "linear: out, in, weight must be 2D");
    
    // 提取张量形状，确立矩阵乘法参数。
    size_t B = in->shape()[0], K = in->shape()[1];
    size_t M = weight->shape()[0];
    
    // 矩阵乘法规则校验：
    // X 的列数 K 必须等于 W^T 的行数。由于我们存储的 W 是 (M, K)，即 W 的第二维长度必须与 in 的第二维长度严格相等。
    ASSERT(weight->shape()[1] == K, "linear: weight second dim must match in second dim");
    // 验证输出张量 out 是否已经分配了正确的形状容纳计算结果。
    ASSERT(out->shape()[0] == B && out->shape()[1] == M, "linear: out shape must be (B, M)");
    if (bias) ASSERT(bias->numel() == M, "linear: bias size must equal M");

    // 提取 bias 的裸指针（Raw Pointer）。若用户未传入 bias，则置为 nullptr，以便底层计算逻辑进行判断。
    std::byte *bias_ptr = bias ? bias->data() : nullptr;

    // 硬件路由分发：基于输出张量所在的设备类型，将底层数据指针分发给对应的硬件计算核心。
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        // 跳转至本文件上方的 CPU 计算逻辑。
        return linear_cpu(out->data(), in->data(), weight->data(), bias_ptr, out->dtype(), B, M, K);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        nvidia::linear(out->data(), in->data(), weight->data(), bias_ptr, out->dtype(), B, M, K);
        return;
#endif
    default:
        // 捕获未注册的设备类型，防止未知行为。
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops