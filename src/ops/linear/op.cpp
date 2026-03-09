#include "op.hpp"
#include "../../utils.hpp"

namespace llaisys::ops {

template<typename T>
void linear_cpu_kernel(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias){
    const T* in_ptr = reinterpret_cast<T*>(in->data());
    const T* weight_ptr = reinterpret_cast<T*>(weight->data());
    const T* bias_ptr = nullptr;
    if(bias && bias->numel() > 0) bias_ptr = reinterpret_cast<T*>(bias->data());
    T* out_ptr = reinterpret_cast<T*>(out->data());

    size_t K = in->shape().back();
    size_t N = weight->shape()[0];

    // 2. 【核心修复】智能计算行数 M
    size_t M;
    if (in->shape().size() == 2) {
        // 情况 A: 单元测试 (2D) [Rows, Hidden]
        M = in->shape()[0]; 
    } else {
        // 情况 B: DeepSeek 推理 (3D) [Batch, Seq, Hidden]
        // 必须把 Batch 和 Seq 乘起来，否则只算第一行！
        M = in->shape()[0] * in->shape()[1];
    }

    const size_t BLOCK_SIZE = 64;

    #pragma omp parallel for collapse(2)
    for(size_t i0 = 0; i0 < M; i0 += BLOCK_SIZE){
        for(size_t j0 = 0; j0 < N; j0 += BLOCK_SIZE){
            
            size_t i_end = std::min(i0 + BLOCK_SIZE, M);
            size_t j_end = std::min(j0 + BLOCK_SIZE, N);

            float block_sum[BLOCK_SIZE][BLOCK_SIZE] = {0.0f};

            for(size_t k0 = 0; k0 < K; k0 += BLOCK_SIZE){
                size_t k_end = std::min(k0 + BLOCK_SIZE, K);

                for(size_t i = i0; i < i_end; ++i){
                    for(size_t j = j0; j < j_end; ++j){
                        float sum = 0.0f;
                        
                        // 让编译器在这里执行连续内存的向量化点积
                        #pragma omp simd reduction(+:sum)
                        for(size_t k = k0; k < k_end; ++k){
                            float x_val = utils::cast<float>(in_ptr[k + i * K]);
                            float y_val = utils::cast<float>(weight_ptr[k + j * K]);
                            sum += x_val * y_val;
                        }
                        block_sum[i - i0][j - j0] += sum;
                    }
                }
            }

            // 写回目标张量
            for(size_t i = i0; i < i_end; ++i){
                for(size_t j = j0; j < j_end; ++j){
                    float final_val = block_sum[i - i0][j - j0];
                    if (bias && bias->numel() > 0) {
                        float b_val = utils::cast<float>(bias_ptr[j]);
                        final_val += b_val;
                    }
                    out_ptr[j + i * N] = utils::cast<T>(final_val);
                }
            }

        }
    }
}

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    in=in->isContiguous()?in:in->contiguous();
    weight = weight->isContiguous() ? weight : weight->contiguous();
    switch (in->dtype()) {
        case LLAISYS_DTYPE_F16:
            linear_cpu_kernel<llaisys::fp16_t>(out,in,weight,bias);
            break;
        case LLAISYS_DTYPE_BF16:
            linear_cpu_kernel<llaisys::bf16_t>(out,in,weight,bias);
            break;
        case LLAISYS_DTYPE_F32:
            linear_cpu_kernel<float>(out,in,weight,bias);
            break;
        default:
            throw std::runtime_error("Not support this dtype!");
    }
}
} // namespace llaisys::ops