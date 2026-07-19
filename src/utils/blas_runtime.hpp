#pragma once
// ============================================================
// 运行时 BLAS 检测与加载 (dlopen)
//
// 在程序首次调用 blas::sgemm() 时自动检测并加载系统上可用的 BLAS 库。
// 优先级: MKL > OpenBLAS > 无 (返回 false, 调用方应 fallback 到手写内核)
//
// 使用方式:
//   if (blas::available()) {
//       blas::sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
//                   M, N, K, 1.0f, A, K, B, K, 0.0f, C, N);
//   }
//
// 此模块不引入编译期依赖, 同一份 .so 可在有/无 BLAS 的机器上运行。
// ============================================================

#include <cstddef>
#include <cstdio>
#include <cstdlib>

#ifndef _WIN32
#include <dlfcn.h>
#endif

namespace llaisys::blas {

// CBLAS 枚举 (与标准 cblas.h 兼容)
enum CBLAS_ORDER     { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113 };

// cblas_sgemm 函数指针类型
using sgemm_fn_t = void (*)(
    CBLAS_ORDER order, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
    int M, int N, int K,
    float alpha, const float* A, int lda,
    const float* B, int ldb,
    float beta, float* C, int ldc);

// 设置线程数的函数指针类型
using set_num_threads_fn_t = void (*)(int);

// 内部状态 (线程安全: 全局初始化一次)
namespace detail {

enum class BlasBackend { NONE, MKL, OPENBLAS };

struct BlasState {
    bool initialized = false;
    bool is_available = false;
    BlasBackend backend = BlasBackend::NONE;
    void* handle = nullptr;
    sgemm_fn_t sgemm_ptr = nullptr;
    set_num_threads_fn_t set_threads_ptr = nullptr;
};

inline BlasState& state() {
    static BlasState s;
    return s;
}

inline void try_load() {
#ifdef _WIN32
    state().initialized = true;
    return;  // Windows 不支持 dlopen
#else
    auto& s = state();
    if (s.initialized) return;
    s.initialized = true;

    // 尝试加载的库列表 (按优先级)
    struct LibCandidate {
        const char* path;
        BlasBackend backend;
        const char* sgemm_sym;
        const char* threads_sym;
    };

    // MKL rt 是单一入口点, 会自动根据 CPU 选择最优内核
    LibCandidate candidates[] = {
        // MKL (优先)
        {"libmkl_rt.so.1",        BlasBackend::MKL,      "cblas_sgemm", "MKL_Set_Num_Threads"},
        {"libmkl_rt.so",          BlasBackend::MKL,      "cblas_sgemm", "MKL_Set_Num_Threads"},
        // OpenBLAS
        {"libopenblas.so.0",      BlasBackend::OPENBLAS, "cblas_sgemm", "openblas_set_num_threads"},
        {"libopenblas.so",        BlasBackend::OPENBLAS, "cblas_sgemm", "openblas_set_num_threads"},
    };

    // MKL 的默认 Intel 线程层 (libiomp5) 在某些服务器上存在 bug,
    // 导致多线程 sgemm 产生错误结果 (累加被重复计算)。
    // 强制使用 GNU 线程层 (libgomp) 避免此问题。
    // 必须在 dlopen 之前设置, 因为 MKL 在加载时读取此环境变量。
    setenv("MKL_THREADING_LAYER", "GNU", 0);  // 0 = 不覆盖用户已设置的值

    for (auto& c : candidates) {
        void* h = dlopen(c.path, RTLD_NOW | RTLD_LOCAL);
        if (!h) continue;

        auto fn = (sgemm_fn_t)dlsym(h, c.sgemm_sym);
        if (!fn) {
            dlclose(h);
            continue;
        }

        s.handle = h;
        s.sgemm_ptr = fn;
        s.backend = c.backend;
        s.is_available = true;

        // 尝试获取线程设置函数 (可选)
        s.set_threads_ptr = (set_num_threads_fn_t)dlsym(h, c.threads_sym);

        const char* name = (c.backend == BlasBackend::MKL) ? "MKL" : "OpenBLAS";
        std::fprintf(stderr, "[llaisys] Runtime BLAS loaded: %s (%s)\n", name, c.path);
        return;
    }

    // 未找到任何 BLAS
    std::fprintf(stderr, "[llaisys] No runtime BLAS found, using AVX2 fallback\n");
#endif
}

} // namespace detail

// 检查是否有可用的 BLAS (首次调用时自动初始化)
inline bool available() {
    if (!detail::state().initialized)
        detail::try_load();
    return detail::state().is_available;
}

// 获取后端名称
inline const char* backend_name() {
    if (!available()) return "none";
    switch (detail::state().backend) {
        case detail::BlasBackend::MKL:      return "MKL";
        case detail::BlasBackend::OPENBLAS: return "OpenBLAS";
        default:                            return "none";
    }
}

// 设置 BLAS 内部线程数
inline void set_num_threads(int n) {
    if (available() && detail::state().set_threads_ptr)
        detail::state().set_threads_ptr(n);
}

// 调用 cblas_sgemm
inline void sgemm(CBLAS_ORDER order, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                   int M, int N, int K,
                   float alpha, const float* A, int lda,
                   const float* B, int ldb,
                   float beta, float* C, int ldc)
{
    detail::state().sgemm_ptr(order, transA, transB,
                              M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

} // namespace llaisys::blas
