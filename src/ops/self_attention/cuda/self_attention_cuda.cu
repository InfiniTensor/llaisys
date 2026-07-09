#include "self_attention_cuda.cuh"

#include "../../../utils.hpp"
#include "../../../utils/check.hpp"

#include <cmath>

namespace llaisys::ops::cuda {

__global__ void self_attention_kernel(void *attn_val, const void *q, const void *k, const void *v,
                                      llaisysDataType_t type, size_t seqlen, size_t totlen, size_t nh, size_t nkvh,
                                      size_t d, size_t dv, float scale) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = seqlen * nh * dv;
    if (idx >= total) {
        return;
    }

    size_t dv_i = idx % dv;
    size_t h = (idx / dv) % nh;
    size_t s = idx / (dv * nh);

    size_t shared = nh / nkvh;
    size_t kvh = h / shared;

    float max_score = -INFINITY;
    for (size_t t = 0; t < totlen; ++t) {
        float score = 0.0f;
        size_t qbase = (s * nh + h) * d;
        size_t kbase = (t * nkvh + kvh) * d;
        for (size_t di = 0; di < d; ++di) {
            score += load_as_float(q, qbase + di, type) * load_as_float(k, kbase + di, type);
        }
        score *= scale;
        if (t > s + totlen - seqlen) {
            score = -INFINITY;
        }
        if (score > max_score) {
            max_score = score;
        }
    }

    float sum_exp = 0.0f;
    for (size_t t = 0; t < totlen; ++t) {
        float score = 0.0f;
        size_t qbase = (s * nh + h) * d;
        size_t kbase = (t * nkvh + kvh) * d;
        for (size_t di = 0; di < d; ++di) {
            score += load_as_float(q, qbase + di, type) * load_as_float(k, kbase + di, type);
        }
        score *= scale;
        if (t > s + totlen - seqlen) {
            score = -INFINITY;
        }
        sum_exp += expf(score - max_score);
    }

    float out_val = 0.0f;
    for (size_t t = 0; t < totlen; ++t) {
        float score = 0.0f;
        size_t qbase = (s * nh + h) * d;
        size_t kbase = (t * nkvh + kvh) * d;
        for (size_t di = 0; di < d; ++di) {
            score += load_as_float(q, qbase + di, type) * load_as_float(k, kbase + di, type);
        }
        score *= scale;
        if (t > s + totlen - seqlen) {
            score = -INFINITY;
        }
        float p = expf(score - max_score) / sum_exp;
        size_t vbase = (t * nkvh + kvh) * dv;
        out_val += p * load_as_float(v, vbase + dv_i, type);
    }

    store_from_float(attn_val, idx, out_val, type);
}

void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t type, size_t seqlen, size_t totlen, size_t nh, size_t nkvh, size_t d, size_t dv,
                    float scale) {
    ASSERT(type == LLAISYS_DTYPE_F32 || type == LLAISYS_DTYPE_F16 || type == LLAISYS_DTYPE_BF16,
           "self_attention cuda only supports F32/F16/BF16");
    size_t total = seqlen * nh * dv;
    dim3 block(128);
    dim3 grid((total + block.x - 1) / block.x);
    self_attention_kernel<<<grid, block>>>(attn_val, q, k, v, type, seqlen, totlen, nh, nkvh, d, dv, scale);
    auto err = cudaGetLastError();
    ASSERT(err == cudaSuccess, "self_attention cuda kernel launch failed");
}

} // namespace llaisys::ops::cuda
