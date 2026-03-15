#include "linear_nvidia.cuh"

#include "../../../utils.hpp"
#include "../../../utils/gpu_utils.hpp"

#include <cublas_v2.h>

#include <cstddef>
#include <stdexcept>

namespace {

inline void cublas_check(cublasStatus_t status, const char *msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(msg);
    }
}

inline cublasHandle_t get_cublas_handle() {
    static thread_local cublasHandle_t handle = []() {
        cublasHandle_t h = nullptr;
        cublas_check(cublasCreate(&h), "cublasCreate failed");
        return h;
    }();
    return handle;
}

template <typename T>
__global__ void add_bias_rowwise_kernel(T *out, const T *bias, size_t M, size_t N) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = M * N;
    for (size_t i = idx; i < total; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const size_t col = i % N;
        out[i] = from_float<T>(to_float(out[i]) + to_float(bias[col]));
    }
}

template <typename T>
inline void launch_add_bias(T *out, const T *bias, size_t M, size_t N) {
    if (bias == nullptr || M == 0 || N == 0) {
        return;
    }
    constexpr int block_size = 256;
    const int grid_size = static_cast<int>(CEIL(M * N, block_size));
    add_bias_rowwise_kernel<<<grid_size, block_size>>>(out, bias, M, N);
}

inline void linear_cublas_f32(float *out,
                              const float *in,
                              const float *weight,
                              const float *bias,
                              size_t M,
                              size_t N,
                              size_t K) {
    cublasHandle_t handle = get_cublas_handle();
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublas_check(cublasSgemm(handle,
                             CUBLAS_OP_T,
                             CUBLAS_OP_N,
                             static_cast<int>(N),
                             static_cast<int>(M),
                             static_cast<int>(K),
                             &alpha,
                             weight,
                             static_cast<int>(K),
                             in,
                             static_cast<int>(K),
                             &beta,
                             out,
                             static_cast<int>(N)),
                 "cublasSgemm failed");
    launch_add_bias(out, bias, M, N);
}

inline void linear_cublas_f16(half *out,
                              const half *in,
                              const half *weight,
                              const half *bias,
                              size_t M,
                              size_t N,
                              size_t K) {
    cublasHandle_t handle = get_cublas_handle();
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t status = cublasGemmEx(handle,
                                         CUBLAS_OP_T,
                                         CUBLAS_OP_N,
                                         static_cast<int>(N),
                                         static_cast<int>(M),
                                         static_cast<int>(K),
                                         &alpha,
                                         weight,
                                         CUDA_R_16F,
                                         static_cast<int>(K),
                                         in,
                                         CUDA_R_16F,
                                         static_cast<int>(K),
                                         &beta,
                                         out,
                                         CUDA_R_16F,
                                         static_cast<int>(N),
                                         CUBLAS_COMPUTE_32F,
                                         CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (status == CUBLAS_STATUS_NOT_SUPPORTED) {
        status = cublasGemmEx(handle,
                              CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              static_cast<int>(N),
                              static_cast<int>(M),
                              static_cast<int>(K),
                              &alpha,
                              weight,
                              CUDA_R_16F,
                              static_cast<int>(K),
                              in,
                              CUDA_R_16F,
                              static_cast<int>(K),
                              &beta,
                              out,
                              CUDA_R_16F,
                              static_cast<int>(N),
                              CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT);
    }
    cublas_check(status, "cublasGemmEx f16 failed");
    launch_add_bias(out, bias, M, N);
}

inline void linear_cublas_bf16(__nv_bfloat16 *out,
                               const __nv_bfloat16 *in,
                               const __nv_bfloat16 *weight,
                               const __nv_bfloat16 *bias,
                               size_t M,
                               size_t N,
                               size_t K) {
    cublasHandle_t handle = get_cublas_handle();
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t status = cublasGemmEx(handle,
                                         CUBLAS_OP_T,
                                         CUBLAS_OP_N,
                                         static_cast<int>(N),
                                         static_cast<int>(M),
                                         static_cast<int>(K),
                                         &alpha,
                                         weight,
                                         CUDA_R_16BF,
                                         static_cast<int>(K),
                                         in,
                                         CUDA_R_16BF,
                                         static_cast<int>(K),
                                         &beta,
                                         out,
                                         CUDA_R_16BF,
                                         static_cast<int>(N),
                                         CUBLAS_COMPUTE_32F,
                                         CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (status == CUBLAS_STATUS_NOT_SUPPORTED) {
        status = cublasGemmEx(handle,
                              CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              static_cast<int>(N),
                              static_cast<int>(M),
                              static_cast<int>(K),
                              &alpha,
                              weight,
                              CUDA_R_16BF,
                              static_cast<int>(K),
                              in,
                              CUDA_R_16BF,
                              static_cast<int>(K),
                              &beta,
                              out,
                              CUDA_R_16BF,
                              static_cast<int>(N),
                              CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT);
    }
    cublas_check(status, "cublasGemmEx bf16 failed");
    launch_add_bias(out, bias, M, N);
}

// Reference-only hand-written kernel retained for review. It is not dispatched.
template <const int BLOCK_SIZE_M,
          const int BLOCK_SIZE_N,
          const int BLOCK_SIZE_K,
          const int THREAD_SIZE_X,
          const int THREAD_SIZE_Y>
__global__ void sgemm_v7_float32(float *__restrict__ out,
                                 const float *__restrict__ in,
                                 const float *__restrict__ weight,
                                 const float *__restrict__ bias,
                                 size_t M,
                                 size_t N,
                                 size_t K) {
    static_assert(BLOCK_SIZE_M == 128 && BLOCK_SIZE_N == 128 && BLOCK_SIZE_K == 8 && THREAD_SIZE_X == 8 && THREAD_SIZE_Y == 8,
                  "v7 is tuned for 128x128x8 tile and 8x8 thread tile.");

    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int thread_x_per_block = BLOCK_SIZE_N / THREAD_SIZE_X;
    const int thread_y_per_block = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int thread_num_per_block = thread_x_per_block * thread_y_per_block;

    const int tid = ty * thread_x_per_block + tx;

    __shared__ float As[2][BLOCK_SIZE_K][BLOCK_SIZE_M];
    __shared__ float Bs[2][BLOCK_SIZE_K][BLOCK_SIZE_N];

    float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0.0f};

    float frag_a[2][THREAD_SIZE_Y];
    float frag_b[2][THREAD_SIZE_X];

    const int ldg_num_a = BLOCK_SIZE_M * BLOCK_SIZE_K / (thread_num_per_block * 4);
    const int ldg_num_b = BLOCK_SIZE_K * BLOCK_SIZE_N / (thread_num_per_block * 4);
    float ldg_a_reg[4 * ldg_num_a];
    float ldg_b_reg[4 * ldg_num_b];

    const int a_load_thread_per_row = BLOCK_SIZE_K / 4;
    const int b_load_thread_per_row = BLOCK_SIZE_K / 4;

    const int a_load_row_start = tid / a_load_thread_per_row;
    const int b_load_row_start = tid / b_load_thread_per_row;
    const int a_load_col = (tid % a_load_thread_per_row) * 4;
    const int b_load_col = (tid % b_load_thread_per_row) * 4;

    const int a_load_row_stride = thread_num_per_block / a_load_thread_per_row;
    const int b_load_row_stride = thread_num_per_block / b_load_thread_per_row;

    const float *A = &in[(BLOCK_SIZE_M * by) * K];
    const float *B = &weight[(BLOCK_SIZE_N * bx) * K];

#pragma unroll
    for (int i = 0; i < BLOCK_SIZE_M; i += a_load_row_stride) {
        const int ldg_index = i / a_load_row_stride * 4;
        const int offset = (a_load_row_start + i) * K + a_load_col;
        STORE_FLOAT4(ldg_a_reg[ldg_index]) = LOAD_FLOAT4(A[offset]);
        As[0][a_load_col][a_load_row_start + i] = ldg_a_reg[ldg_index];
        As[0][a_load_col + 1][a_load_row_start + i] = ldg_a_reg[ldg_index + 1];
        As[0][a_load_col + 2][a_load_row_start + i] = ldg_a_reg[ldg_index + 2];
        As[0][a_load_col + 3][a_load_row_start + i] = ldg_a_reg[ldg_index + 3];
    }

#pragma unroll
    for (int i = 0; i < BLOCK_SIZE_N; i += b_load_row_stride) {
        const int ldg_index = i / b_load_row_stride * 4;
        const int offset = (b_load_row_start + i) * K + b_load_col;
        STORE_FLOAT4(ldg_b_reg[ldg_index]) = LOAD_FLOAT4(B[offset]);
        Bs[0][b_load_col][b_load_row_start + i] = ldg_b_reg[ldg_index];
        Bs[0][b_load_col + 1][b_load_row_start + i] = ldg_b_reg[ldg_index + 1];
        Bs[0][b_load_col + 2][b_load_row_start + i] = ldg_b_reg[ldg_index + 2];
        Bs[0][b_load_col + 3][b_load_row_start + i] = ldg_b_reg[ldg_index + 3];
    }
    __syncthreads();

    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int a_tile_index = warp_id / 2 * 16 + lane_id / 8 * 4;
    const int b_tile_index = warp_id % 2 * 32 + lane_id % 8 * 4;

    STORE_FLOAT4(frag_a[0][0]) = LOAD_FLOAT4(As[0][0][a_tile_index]);
    STORE_FLOAT4(frag_a[0][4]) = LOAD_FLOAT4(As[0][0][a_tile_index + BLOCK_SIZE_M / 2]);
    STORE_FLOAT4(frag_b[0][0]) = LOAD_FLOAT4(Bs[0][0][b_tile_index]);
    STORE_FLOAT4(frag_b[0][4]) = LOAD_FLOAT4(Bs[0][0][b_tile_index + BLOCK_SIZE_N / 2]);

    int write_stage_idx = 1;
    int tile_idx = 0;
    do {
        tile_idx += BLOCK_SIZE_K;
        if (tile_idx < K) {
#pragma unroll
            for (int i = 0; i < BLOCK_SIZE_M; i += a_load_row_stride) {
                const int ldg_index = i / a_load_row_stride * 4;
                const int offset = (a_load_row_start + i) * K + (a_load_col + tile_idx);
                STORE_FLOAT4(ldg_a_reg[ldg_index]) = LOAD_FLOAT4(A[offset]);
            }
#pragma unroll
            for (int i = 0; i < BLOCK_SIZE_N; i += b_load_row_stride) {
                const int ldg_index = i / b_load_row_stride * 4;
                const int offset = (b_load_row_start + i) * K + (b_load_col + tile_idx);
                STORE_FLOAT4(ldg_b_reg[ldg_index]) = LOAD_FLOAT4(B[offset]);
            }
        }

        const int load_stage_idx = write_stage_idx ^ 1;

#pragma unroll
        for (int j = 0; j < BLOCK_SIZE_K - 1; ++j) {
            STORE_FLOAT4(frag_a[(j + 1) % 2][0]) = LOAD_FLOAT4(As[load_stage_idx][j + 1][a_tile_index]);
            STORE_FLOAT4(frag_a[(j + 1) % 2][4]) = LOAD_FLOAT4(As[load_stage_idx][j + 1][a_tile_index + BLOCK_SIZE_M / 2]);
            STORE_FLOAT4(frag_b[(j + 1) % 2][0]) = LOAD_FLOAT4(Bs[load_stage_idx][j + 1][b_tile_index]);
            STORE_FLOAT4(frag_b[(j + 1) % 2][4]) = LOAD_FLOAT4(Bs[load_stage_idx][j + 1][b_tile_index + BLOCK_SIZE_N / 2]);

#pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
#pragma unroll
                for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                    accum[thread_y][thread_x] += frag_a[j % 2][thread_y] * frag_b[j % 2][thread_x];
                }
            }
        }

        if (tile_idx < K) {
#pragma unroll
            for (int i = 0; i < BLOCK_SIZE_M; i += a_load_row_stride) {
                const int ldg_index = i / a_load_row_stride * 4;
                As[write_stage_idx][a_load_col][a_load_row_start + i] = ldg_a_reg[ldg_index];
                As[write_stage_idx][a_load_col + 1][a_load_row_start + i] = ldg_a_reg[ldg_index + 1];
                As[write_stage_idx][a_load_col + 2][a_load_row_start + i] = ldg_a_reg[ldg_index + 2];
                As[write_stage_idx][a_load_col + 3][a_load_row_start + i] = ldg_a_reg[ldg_index + 3];
            }
#pragma unroll
            for (int i = 0; i < BLOCK_SIZE_N; i += b_load_row_stride) {
                const int ldg_index = i / b_load_row_stride * 4;
                Bs[write_stage_idx][b_load_col][b_load_row_start + i] = ldg_b_reg[ldg_index];
                Bs[write_stage_idx][b_load_col + 1][b_load_row_start + i] = ldg_b_reg[ldg_index + 1];
                Bs[write_stage_idx][b_load_col + 2][b_load_row_start + i] = ldg_b_reg[ldg_index + 2];
                Bs[write_stage_idx][b_load_col + 3][b_load_row_start + i] = ldg_b_reg[ldg_index + 3];
            }
            __syncthreads();
            write_stage_idx ^= 1;
        }

        STORE_FLOAT4(frag_a[0][0]) = LOAD_FLOAT4(As[load_stage_idx ^ 1][0][a_tile_index]);
        STORE_FLOAT4(frag_a[0][4]) = LOAD_FLOAT4(As[load_stage_idx ^ 1][0][a_tile_index + BLOCK_SIZE_M / 2]);
        STORE_FLOAT4(frag_b[0][0]) = LOAD_FLOAT4(Bs[load_stage_idx ^ 1][0][b_tile_index]);
        STORE_FLOAT4(frag_b[0][4]) = LOAD_FLOAT4(Bs[load_stage_idx ^ 1][0][b_tile_index + BLOCK_SIZE_N / 2]);

#pragma unroll
        for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
#pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                accum[thread_y][thread_x] += frag_a[1][thread_y] * frag_b[1][thread_x];
            }
        }
    } while (tile_idx < K);

    const int c_block_row = a_tile_index;
    const int c_block_col = b_tile_index;

    for (int i = 0; i < 4; i++) {
        const int row = BLOCK_SIZE_M * by + c_block_row + i;
        const int col = BLOCK_SIZE_N * bx + c_block_col;
        float4 c_val;
        c_val.x = accum[i][0];
        c_val.y = accum[i][1];
        c_val.z = accum[i][2];
        c_val.w = accum[i][3];
        if (bias != nullptr) {
            c_val.x += bias[col];
            c_val.y += bias[col + 1];
            c_val.z += bias[col + 2];
            c_val.w += bias[col + 3];
        }
        STORE_FLOAT4(out[row * N + col]) = c_val;
    }
    for (int i = 0; i < 4; i++) {
        const int row = BLOCK_SIZE_M * by + c_block_row + i;
        const int col = BLOCK_SIZE_N * bx + c_block_col + BLOCK_SIZE_N / 2;
        float4 c_val;
        c_val.x = accum[i][4];
        c_val.y = accum[i][5];
        c_val.z = accum[i][6];
        c_val.w = accum[i][7];
        if (bias != nullptr) {
            c_val.x += bias[col];
            c_val.y += bias[col + 1];
            c_val.z += bias[col + 2];
            c_val.w += bias[col + 3];
        }
        STORE_FLOAT4(out[row * N + col]) = c_val;
    }
    for (int i = 0; i < 4; i++) {
        const int row = BLOCK_SIZE_M * by + c_block_row + BLOCK_SIZE_M / 2 + i;
        const int col = BLOCK_SIZE_N * bx + c_block_col;
        float4 c_val;
        c_val.x = accum[i + 4][0];
        c_val.y = accum[i + 4][1];
        c_val.z = accum[i + 4][2];
        c_val.w = accum[i + 4][3];
        if (bias != nullptr) {
            c_val.x += bias[col];
            c_val.y += bias[col + 1];
            c_val.z += bias[col + 2];
            c_val.w += bias[col + 3];
        }
        STORE_FLOAT4(out[row * N + col]) = c_val;
    }
    for (int i = 0; i < 4; i++) {
        const int row = BLOCK_SIZE_M * by + c_block_row + BLOCK_SIZE_M / 2 + i;
        const int col = BLOCK_SIZE_N * bx + c_block_col + BLOCK_SIZE_N / 2;
        float4 c_val;
        c_val.x = accum[i + 4][4];
        c_val.y = accum[i + 4][5];
        c_val.z = accum[i + 4][6];
        c_val.w = accum[i + 4][7];
        if (bias != nullptr) {
            c_val.x += bias[col];
            c_val.y += bias[col + 1];
            c_val.z += bias[col + 2];
            c_val.w += bias[col + 3];
        }
        STORE_FLOAT4(out[row * N + col]) = c_val;
    }
}

} // namespace

namespace llaisys::ops::nvidia {

void linear(std::byte *out,
            const std::byte *in,
            const std::byte *weight,
            const std::byte *bias,
            llaisysDataType_t type,
            size_t M,
            size_t N,
            size_t K) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        linear_cublas_f32(reinterpret_cast<float *>(out),
                          reinterpret_cast<const float *>(in),
                          reinterpret_cast<const float *>(weight),
                          reinterpret_cast<const float *>(bias),
                          M,
                          N,
                          K);
        break;
    case LLAISYS_DTYPE_F16:
        linear_cublas_f16(reinterpret_cast<half *>(out),
                          reinterpret_cast<const half *>(in),
                          reinterpret_cast<const half *>(weight),
                          reinterpret_cast<const half *>(bias),
                          M,
                          N,
                          K);
        break;
    case LLAISYS_DTYPE_BF16:
        linear_cublas_bf16(reinterpret_cast<__nv_bfloat16 *>(out),
                           reinterpret_cast<const __nv_bfloat16 *>(in),
                           reinterpret_cast<const __nv_bfloat16 *>(weight),
                           reinterpret_cast<const __nv_bfloat16 *>(bias),
                           M,
                           N,
                           K);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    CUDA_CHECK(cudaGetLastError());
}

} // namespace llaisys::ops::nvidia
