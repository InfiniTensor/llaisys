#include "ops_metax.cuh"

#include "../../core/context/context.hpp"
#include "../../device/metax/metax_resource.cuh"
#include "../../device/metax/metax_utils.cuh"
#include "../argmax/cpu/argmax_cpu.hpp"
#include "../self_attention/cpu/self_attention_cpu.hpp"

#include <algorithm>
#include <vector>

namespace llaisys::ops::metax {
namespace {
using ::half;
using ::maca_bfloat16;

mcStream_t current_stream() {
    // 当前公开算子接口仍然是同步语义，统一落默认 stream，
    // 避免和 Python 侧 torch.cuda 的自定义 stream 发生可见性竞态。
    return nullptr;
}

template <typename T>
__global__ void add_kernel(T *out, const T *lhs, const T *rhs, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) {
        return;
    }
    out[idx] = llaisys::device::metax::from_float_device<T>(
        llaisys::device::metax::to_float_device(lhs[idx]) + llaisys::device::metax::to_float_device(rhs[idx]));
}

template <typename T>
__global__ void swiglu_kernel(T *out, const T *gate, const T *up, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) {
        return;
    }
    float gate_val = llaisys::device::metax::to_float_device(gate[idx]);
    float up_val = llaisys::device::metax::to_float_device(up[idx]);
    float silu = gate_val / (1.0f + expf(-gate_val));
    out[idx] = llaisys::device::metax::from_float_device<T>(up_val * silu);
}

template <typename T>
__global__ void embedding_kernel(T *out, const int64_t *index, const T *weight, size_t num_tokens, size_t hidden_size) {
    size_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = num_tokens * hidden_size;
    if (linear >= total) {
        return;
    }
    size_t token = linear / hidden_size;
    size_t hidden = linear % hidden_size;
    int64_t vocab_idx = index[token];
    out[linear] = weight[vocab_idx * hidden_size + hidden];
}

template <typename T>
__global__ void rope_kernel(T *out, const T *in, const int64_t *pos_ids, float theta, size_t n_head, size_t head_dim, size_t half_dim, size_t total_pairs) {
    size_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= total_pairs) {
        return;
    }
    size_t j = linear % half_dim;
    size_t head_index = linear / half_dim;
    size_t h = head_index % n_head;
    size_t s = head_index / n_head;

    size_t base = s * n_head * head_dim + h * head_dim;
    float pos = static_cast<float>(pos_ids[s]);
    float angle = pos / powf(theta, (2.0f * static_cast<float>(j)) / static_cast<float>(head_dim));
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    float a = llaisys::device::metax::to_float_device(in[base + j]);
    float b = llaisys::device::metax::to_float_device(in[base + j + half_dim]);
    out[base + j] = llaisys::device::metax::from_float_device<T>(a * cos_val - b * sin_val);
    out[base + j + half_dim] = llaisys::device::metax::from_float_device<T>(b * cos_val + a * sin_val);
}

template <typename T>
__global__ void bias_kernel(T *out, const T *bias, size_t M, size_t N) {
    size_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = M * N;
    if (linear >= total) {
        return;
    }
    size_t col = linear % N;
    float value = llaisys::device::metax::to_float_device(out[linear]) + llaisys::device::metax::to_float_device(bias[col]);
    out[linear] = llaisys::device::metax::from_float_device<T>(value);
}

template <typename T>
__global__ void rms_norm_kernel(T *out, const T *in, const T *weight, float eps, size_t hidden_size) {
    size_t row = blockIdx.x;
    size_t tid = threadIdx.x;
    extern __shared__ float shared[];
    float sum_sq = 0.0f;

    const T *row_in = in + row * hidden_size;
    T *row_out = out + row * hidden_size;

    for (size_t col = tid; col < hidden_size; col += blockDim.x) {
        float value = llaisys::device::metax::to_float_device(row_in[col]);
        sum_sq += value * value;
    }
    shared[tid] = sum_sq;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    float inv_rms = rsqrtf(shared[0] / static_cast<float>(hidden_size) + eps);
    for (size_t col = tid; col < hidden_size; col += blockDim.x) {
        float value = llaisys::device::metax::to_float_device(row_in[col]);
        float scale = llaisys::device::metax::to_float_device(weight[col]);
        row_out[col] = llaisys::device::metax::from_float_device<T>(value * inv_rms * scale);
    }
}

template <typename KernelFunc, typename... Args>
void launch_1d(KernelFunc kernel, size_t numel, Args... args) {
    constexpr int THREADS = 256;
    int blocks = static_cast<int>((numel + THREADS - 1) / THREADS);
    kernel<<<blocks, THREADS, 0, current_stream()>>>(args..., numel);
    METAX_CHECK(mcGetLastError());
}

template <typename T>
void add_impl(std::byte *c, const std::byte *a, const std::byte *b, size_t numel) {
    launch_1d(add_kernel<T>, numel, reinterpret_cast<T *>(c), reinterpret_cast<const T *>(a), reinterpret_cast<const T *>(b));
}

template <typename T>
void swiglu_impl(std::byte *out, const std::byte *gate, const std::byte *up, size_t numel) {
    launch_1d(swiglu_kernel<T>, numel, reinterpret_cast<T *>(out), reinterpret_cast<const T *>(gate), reinterpret_cast<const T *>(up));
}

template <typename T>
void embedding_impl(std::byte *out, const std::byte *index, const std::byte *weight, size_t num_tokens, size_t hidden_size) {
    size_t total = num_tokens * hidden_size;
    constexpr int THREADS = 256;
    int blocks = static_cast<int>((total + THREADS - 1) / THREADS);
    embedding_kernel<T><<<blocks, THREADS, 0, current_stream()>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const int64_t *>(index),
        reinterpret_cast<const T *>(weight),
        num_tokens,
        hidden_size);
    METAX_CHECK(mcGetLastError());
}

template <typename T>
void rope_impl(std::byte *out, const std::byte *in, const std::byte *pos_ids, float theta, size_t seq_len, size_t n_head, size_t head_dim) {
    size_t half_dim = head_dim / 2;
    size_t total_pairs = seq_len * n_head * half_dim;
    constexpr int THREADS = 256;
    int blocks = static_cast<int>((total_pairs + THREADS - 1) / THREADS);
    rope_kernel<T><<<blocks, THREADS, 0, current_stream()>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(in),
        reinterpret_cast<const int64_t *>(pos_ids),
        theta,
        n_head,
        head_dim,
        half_dim,
        total_pairs);
    METAX_CHECK(mcGetLastError());
}

template <typename T>
void rms_norm_impl(std::byte *out, const std::byte *in, const std::byte *weight, float eps, size_t num_rows, size_t hidden_size) {
    constexpr int THREADS = 256;
    rms_norm_kernel<T><<<static_cast<int>(num_rows), THREADS, THREADS * sizeof(float), current_stream()>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(in),
        reinterpret_cast<const T *>(weight),
        eps,
        hidden_size);
    METAX_CHECK(mcGetLastError());
}

template <typename T>
void maybe_add_bias(std::byte *out, const std::byte *bias, size_t M, size_t N) {
    if (bias == nullptr) {
        return;
    }
    size_t total = M * N;
    constexpr int THREADS = 256;
    int blocks = static_cast<int>((total + THREADS - 1) / THREADS);
    bias_kernel<T><<<blocks, THREADS, 0, current_stream()>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(bias),
        M,
        N);
    METAX_CHECK(mcGetLastError());
}

void host_fallback_argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, size_t numel, llaisysDataType_t dtype, llaisysDataType_t idx_dtype) {
    auto &runtime = llaisys::core::context().runtime();
    auto host_vals = runtime.allocateHostStorage(numel * llaisys::utils::dsize(dtype));
    auto host_idx = runtime.allocateHostStorage(llaisys::utils::dsize(idx_dtype));
    auto host_val = runtime.allocateHostStorage(llaisys::utils::dsize(dtype));
    runtime.api()->memcpy_sync(host_vals->memory(), vals, numel * llaisys::utils::dsize(dtype), LLAISYS_MEMCPY_D2H);
    llaisys::ops::cpu::argmax(host_idx->memory(), host_val->memory(), host_vals->memory(), numel, dtype, idx_dtype);
    runtime.api()->memcpy_sync(max_idx, host_idx->memory(), llaisys::utils::dsize(idx_dtype), LLAISYS_MEMCPY_H2D);
    runtime.api()->memcpy_sync(max_val, host_val->memory(), llaisys::utils::dsize(dtype), LLAISYS_MEMCPY_H2D);
}

void host_fallback_self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v, size_t seqlen, size_t total_len, size_t nhead, size_t nkvhead, size_t d, size_t dv, float scale, llaisysDataType_t dtype) {
    auto &runtime = llaisys::core::context().runtime();
    size_t q_bytes = seqlen * nhead * d * llaisys::utils::dsize(dtype);
    size_t k_bytes = total_len * nkvhead * d * llaisys::utils::dsize(dtype);
    size_t v_bytes = total_len * nkvhead * dv * llaisys::utils::dsize(dtype);
    size_t out_bytes = seqlen * nhead * dv * llaisys::utils::dsize(dtype);

    auto host_q = runtime.allocateHostStorage(q_bytes);
    auto host_k = runtime.allocateHostStorage(k_bytes);
    auto host_v = runtime.allocateHostStorage(v_bytes);
    auto host_out = runtime.allocateHostStorage(out_bytes);
    runtime.api()->memcpy_sync(host_q->memory(), q, q_bytes, LLAISYS_MEMCPY_D2H);
    runtime.api()->memcpy_sync(host_k->memory(), k, k_bytes, LLAISYS_MEMCPY_D2H);
    runtime.api()->memcpy_sync(host_v->memory(), v, v_bytes, LLAISYS_MEMCPY_D2H);
    llaisys::ops::cpu::self_attention(host_out->memory(), host_q->memory(), host_k->memory(), host_v->memory(), seqlen, total_len, nhead, nkvhead, d, dv, scale, dtype);
    runtime.api()->memcpy_sync(attn_val, host_out->memory(), out_bytes, LLAISYS_MEMCPY_H2D);
}
} // namespace

void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return add_impl<float>(c, a, b, numel);
    case LLAISYS_DTYPE_F16:
        return add_impl<half>(c, a, b, numel);
    case LLAISYS_DTYPE_BF16:
        return add_impl<maca_bfloat16>(c, a, b, numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, size_t numel, llaisysDataType_t dtype, llaisysDataType_t idx_dtype) {
    host_fallback_argmax(max_idx, max_val, vals, numel, dtype, idx_dtype);
}

void embedding(std::byte *out, const std::byte *index, const std::byte *weight, size_t num_tokens, size_t /*vocab_size*/, size_t hidden_size, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return embedding_impl<float>(out, index, weight, num_tokens, hidden_size);
    case LLAISYS_DTYPE_F16:
        return embedding_impl<half>(out, index, weight, num_tokens, hidden_size);
    case LLAISYS_DTYPE_BF16:
        return embedding_impl<maca_bfloat16>(out, index, weight, num_tokens, hidden_size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, size_t M, size_t K, size_t N, llaisysDataType_t dtype) {
    auto &runtime = llaisys::core::context().runtime();
    auto handle = llaisys::device::metax::get_mcblas_handle(runtime.deviceId(), current_stream());
    const float alpha = 1.0f;
    const float beta = 0.0f;
    auto maca_dtype = llaisys::device::metax::to_maca_dtype(dtype);
    MCBLAS_CHECK(mcblasSetPointerMode(handle, MCBLAS_POINTER_MODE_HOST));
    // MetaX 上大矩阵线性层和 torch 对照会出现微小数值漂移，
    // 统一切到 GemmEx + pedantic math，优先保证测试一致性。
    MCBLAS_CHECK(mcblasSetMathMode(handle, MCBLAS_PEDANTIC_MATH));
    MCBLAS_CHECK(mcblasGemmEx(
        handle,
        MCBLAS_OP_T,
        MCBLAS_OP_N,
        static_cast<int>(N),
        static_cast<int>(M),
        static_cast<int>(K),
        &alpha,
        weight,
        maca_dtype,
        static_cast<int>(K),
        in,
        maca_dtype,
        static_cast<int>(K),
        &beta,
        out,
        maca_dtype,
        static_cast<int>(N),
        dtype == LLAISYS_DTYPE_F32 ? MCBLAS_COMPUTE_32F_PEDANTIC : MCBLAS_COMPUTE_32F,
        MCBLAS_GEMM_DEFAULT));

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return maybe_add_bias<float>(out, bias, M, N);
    case LLAISYS_DTYPE_F16:
        return maybe_add_bias<half>(out, bias, M, N);
    case LLAISYS_DTYPE_BF16:
        return maybe_add_bias<maca_bfloat16>(out, bias, M, N);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, float eps, size_t num_rows, size_t hidden_size, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_impl<float>(out, in, weight, eps, num_rows, hidden_size);
    case LLAISYS_DTYPE_F16:
        return rms_norm_impl<half>(out, in, weight, eps, num_rows, hidden_size);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_impl<maca_bfloat16>(out, in, weight, eps, num_rows, hidden_size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, float theta, size_t seq_len, size_t n_head, size_t head_dim, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rope_impl<float>(out, in, pos_ids, theta, seq_len, n_head, head_dim);
    case LLAISYS_DTYPE_F16:
        return rope_impl<half>(out, in, pos_ids, theta, seq_len, n_head, head_dim);
    case LLAISYS_DTYPE_BF16:
        return rope_impl<maca_bfloat16>(out, in, pos_ids, theta, seq_len, n_head, head_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v, size_t seqlen, size_t total_len, size_t nhead, size_t nkvhead, size_t d, size_t dv, float scale, llaisysDataType_t dtype) {
    host_fallback_self_attention(attn_val, q, k, v, seqlen, total_len, nhead, nkvhead, d, dv, scale, dtype);
}

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, size_t numel, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return swiglu_impl<float>(out, gate, up, numel);
    case LLAISYS_DTYPE_F16:
        return swiglu_impl<half>(out, gate, up, numel);
    case LLAISYS_DTYPE_BF16:
        return swiglu_impl<maca_bfloat16>(out, gate, up, numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::metax
