#include "self_attention_nvidia.cuh"

#include "../../nvidia/nvidia_common.cuh"

namespace {

constexpr size_t kMaxValueDim = 1024;

template <typename T>
__global__ void self_attention_kernel(T *attn_val,
                                      const T *q,
                                      const T *k,
                                      const T *v,
                                      float scale,
                                      int64_t sq,
                                      int64_t nh,
                                      int64_t d,
                                      int64_t sk,
                                      int64_t nh_kv,
                                      int64_t dv) {
    if (threadIdx.x != 0) {
        return;
    }

    const int64_t work_idx = static_cast<int64_t>(blockIdx.x);
    const int64_t qi = work_idx / nh;
    const int64_t h = work_idx % nh;
    const int64_t n_rep = nh / nh_kv;
    const int64_t h_kv = h / n_rep;
    const int64_t q_abs_pos = sk - sq + qi;
    const int64_t valid_len = q_abs_pos + 1 < sk ? q_abs_pos + 1 : sk;
    const T *q_vec = q + (qi * nh + h) * d;
    T *out_ptr = attn_val + (qi * nh + h) * dv;

    float out_local[kMaxValueDim];
    for (int64_t l = 0; l < dv; ++l) {
        out_local[l] = 0.0f;
    }

    if (valid_len <= 0) {
        for (int64_t l = 0; l < dv; ++l) {
            out_ptr[l] = llaisys::device::nvidia::cuda_utils::fromFloat<T>(0.0f);
        }
        return;
    }

    float max_score = -1.0e30f;
    float sum_exp = 0.0f;
    for (int64_t j = 0; j < valid_len; ++j) {
        const T *k_vec = k + (j * nh_kv + h_kv) * d;
        const T *v_vec = v + (j * nh_kv + h_kv) * dv;

        float dot = 0.0f;
        for (int64_t l = 0; l < d; ++l) {
            dot += llaisys::device::nvidia::cuda_utils::toFloat<T>(q_vec[l]) *
                   llaisys::device::nvidia::cuda_utils::toFloat<T>(k_vec[l]);
        }
        const float score = dot * scale;

        if (score > max_score) {
            const float rescale = expf(max_score - score);
            if (sum_exp > 0.0f) {
                for (int64_t l = 0; l < dv; ++l) {
                    out_local[l] *= rescale;
                }
            }
            sum_exp = sum_exp * rescale + 1.0f;
            max_score = score;
            for (int64_t l = 0; l < dv; ++l) {
                out_local[l] += llaisys::device::nvidia::cuda_utils::toFloat<T>(v_vec[l]);
            }
        } else {
            const float weight = expf(score - max_score);
            sum_exp += weight;
            for (int64_t l = 0; l < dv; ++l) {
                out_local[l] += weight * llaisys::device::nvidia::cuda_utils::toFloat<T>(v_vec[l]);
            }
        }
    }

    const float inv_sum = 1.0f / sum_exp;
    for (int64_t l = 0; l < dv; ++l) {
        out_ptr[l] = llaisys::device::nvidia::cuda_utils::fromFloat<T>(out_local[l] * inv_sum);
    }
}

template <typename T>
void self_attention_impl(std::byte *attn_val,
                         const std::byte *q,
                         const std::byte *k,
                         const std::byte *v,
                         float scale,
                         const std::vector<size_t> &q_shape,
                         const std::vector<size_t> &k_shape,
                         const std::vector<size_t> &v_shape) {
    const int64_t sq = static_cast<int64_t>(q_shape[0]);
    const int64_t nh = static_cast<int64_t>(q_shape[1]);
    const int64_t d = static_cast<int64_t>(q_shape[2]);
    const int64_t sk = static_cast<int64_t>(k_shape[0]);
    const int64_t nh_kv = static_cast<int64_t>(k_shape[1]);
    const int64_t dv = static_cast<int64_t>(v_shape[2]);

    CHECK_ARGUMENT(static_cast<size_t>(dv) <= kMaxValueDim, "NVIDIA self_attention only supports value head dim up to 1024 in this implementation.");

    self_attention_kernel<<<static_cast<unsigned int>(sq * nh), 1, 0, llaisys::ops::nvidia::current_stream()>>>(
        reinterpret_cast<T *>(attn_val),
        reinterpret_cast<const T *>(q),
        reinterpret_cast<const T *>(k),
        reinterpret_cast<const T *>(v),
        scale,
        sq,
        nh,
        d,
        sk,
        nh_kv,
        dv);
    LLAISYS_CUDA_CHECK(cudaGetLastError());
}

} // namespace

namespace llaisys::ops::nvidia {

void self_attention(std::byte *attn_val,
                    const std::byte *q,
                    const std::byte *k,
                    const std::byte *v,
                    float scale,
                    llaisysDataType_t dtype,
                    const std::vector<size_t> &q_shape,
                    const std::vector<size_t> &k_shape,
                    const std::vector<size_t> &v_shape) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return self_attention_impl<float>(attn_val, q, k, v, scale, q_shape, k_shape, v_shape);
    case LLAISYS_DTYPE_F16:
        return self_attention_impl<llaisys::fp16_t>(attn_val, q, k, v, scale, q_shape, k_shape, v_shape);
    case LLAISYS_DTYPE_BF16:
        return self_attention_impl<llaisys::bf16_t>(attn_val, q, k, v, scale, q_shape, k_shape, v_shape);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia
