#include "op.hpp"
#include <cmath>

namespace llaisys::ops {
template<typename T>
void rope_cpu_kernel(tensor_t out, tensor_t in, tensor_t pos_ids, float theta){
    const int64_t* pos_ptr_debug = reinterpret_cast<const int64_t*>(pos_ids->data());
    // 只打印第一个位置 ID
    static int print_count = 0;
    if (print_count < 5) {
        printf("[DEBUG RoPE] pos_ids[0]: %ld, theta: %f\n", pos_ptr_debug[0], theta);
        print_count++;
    }
    T* out_ptr = reinterpret_cast<T*>(out->data());
    const T* in_ptr = reinterpret_cast<const T*>(in->data());
    const int64_t* pos_ptr = reinterpret_cast<const int64_t*>(pos_ids->data());

    const auto& in_shape = in->shape();
    const auto& pos_shape = pos_ids->shape();

    // -----------------------------
    // 模式 1：单元测试 / 通用 3D 形式
    //   in  : [seq_len, n_heads, head_dim]
    //   pos : [seq_len]
    // 与 test/ops/rope.py 中 torch_rope 完全对齐
    // 这里严格按 Tensor 的 strides 访问，避免对内存布局做任何假设
    // -----------------------------
    if (in_shape.size() == 3 && pos_shape.size() == 1 && pos_ids->numel() == in_shape[0]) {
        size_t seq_len = in_shape[0];
        size_t n_heads = in_shape[1];
        size_t head_dim = in_shape[2];

        const auto& in_strides = in->strides();
        const auto& out_strides = out->strides();
        ptrdiff_t s_in_0 = in_strides[0];
        ptrdiff_t s_in_1 = in_strides[1];
        ptrdiff_t s_in_2 = in_strides[2];
        ptrdiff_t s_out_0 = out_strides[0];
        ptrdiff_t s_out_1 = out_strides[1];
        ptrdiff_t s_out_2 = out_strides[2];

        for (size_t i = 0; i < seq_len; ++i) {
            float p_i = utils::cast<float>(pos_ptr[i]);
            for (size_t h = 0; h < n_heads; ++h) {
                for (size_t k = 0; k < head_dim / 2; ++k) {
                    // 计算输入/输出索引，完全依赖 strides
                    size_t idx_a_in = i * s_in_0 + h * s_in_1 + k * s_in_2;
                    size_t idx_b_in = i * s_in_0 + h * s_in_1 + (k + head_dim / 2) * s_in_2;
                    size_t idx_a_out = i * s_out_0 + h * s_out_1 + k * s_out_2;
                    size_t idx_b_out = i * s_out_0 + h * s_out_1 + (k + head_dim / 2) * s_out_2;

                    float theta_in = p_i / std::pow(theta, 2.0f * k / head_dim);

                    float a_in = utils::cast<float>(in_ptr[idx_a_in]);
                    float b_in = utils::cast<float>(in_ptr[idx_b_in]);

                    float cos_t = std::cos(theta_in);
                    float sin_t = std::sin(theta_in);

                    T a_out = utils::cast<T>(a_in * cos_t - b_in * sin_t);
                    T b_out = utils::cast<T>(b_in * cos_t + a_in * sin_t);

                    out_ptr[idx_a_out] = a_out;
                    out_ptr[idx_b_out] = b_out;
                }
            }
        }
        return;
    }

    // -----------------------------
    // 模式 2：Qwen2 / DeepSeek 推理路径
    //   in  : [batch, seq_len, hidden]
    //   pos : [1, seq_len] 或 [seq_len]
    // -----------------------------
    size_t N = in_shape[0]; // batch
    size_t M = in_shape[1]; // seq_len
    size_t D = in_shape[2]; // hidden

    size_t n_pos = pos_ids->numel();

    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < M; j++) {
            size_t flat_idx = i * M + j;
            float p_i;
            if (n_pos == M) {
                p_i = utils::cast<float>(pos_ptr[j]);
            } else if (n_pos == N * M) {
                p_i = utils::cast<float>(pos_ptr[flat_idx]);
            } else {
                p_i = utils::cast<float>(pos_ptr[flat_idx % n_pos]);
            }

            size_t base_offset = i * (M * D) + j * D;

            if (D == 1536 || D == 256) {
                size_t head_dim = 128;
                size_t n_heads = D / 128;

                for (size_t h = 0; h < n_heads; h++) {
                    for (size_t k = 0; k < head_dim / 2; k++) {
                        size_t offset = base_offset + h * head_dim;
                        size_t idx_a = offset + k;
                        size_t idx_b = offset + k + head_dim / 2;

                        float theta_in = p_i / std::pow(theta, 2.0f * k / head_dim);

                        float a_in = utils::cast<float>(in_ptr[idx_a]);
                        float b_in = utils::cast<float>(in_ptr[idx_b]);

                        float cos_t = std::cos(theta_in);
                        float sin_t = std::sin(theta_in);

                        T a_out = utils::cast<T>(a_in * cos_t - b_in * sin_t);
                        T b_out = utils::cast<T>(b_in * cos_t + a_in * sin_t);

                        out_ptr[idx_a] = a_out;
                        out_ptr[idx_b] = b_out;
                    }
                }
            } else {
                for (size_t k = 0; k < D / 2; k++) {
                    size_t idx_a = base_offset + k;
                    size_t idx_b = base_offset + k + D / 2;

                    float theta_in = p_i / std::pow(theta, 2.0f * k / D);

                    float a_in = utils::cast<float>(in_ptr[idx_a]);
                    float b_in = utils::cast<float>(in_ptr[idx_b]);

                    float cos_t = std::cos(theta_in);
                    float sin_t = std::sin(theta_in);

                    T a_out = utils::cast<T>(a_in * cos_t - b_in * sin_t);
                    T b_out = utils::cast<T>(b_in * cos_t + a_in * sin_t);

                    out_ptr[idx_a] = a_out;
                    out_ptr[idx_b] = b_out;
                }
            }
        }
    }
}

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    tensor_t cin = in->isContiguous() ? in : in->contiguous();
    switch (in->dtype()) {
        case LLAISYS_DTYPE_F16:
            rope_cpu_kernel<llaisys::fp16_t>(out,cin,pos_ids,theta);
            break;
        case LLAISYS_DTYPE_BF16:
            rope_cpu_kernel<llaisys::bf16_t>(out,cin,pos_ids,theta);
            break;
        case LLAISYS_DTYPE_F32:
            rope_cpu_kernel<float>(out,cin,pos_ids,theta);
            break;
        default:
            throw std::runtime_error("Not support this dtype!");
    }
}
} // namespace llaisys::ops