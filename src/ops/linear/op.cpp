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
        // 你的原版逻辑，保持不变
        M = in->shape()[0]; 
    } else {
        // 情况 B: DeepSeek 推理 (3D) [Batch, Seq, Hidden]
        // 必须把 Batch 和 Seq 乘起来，否则只算第一行！
        M = in->shape()[0] * in->shape()[1];
    }

    // 3. 矩阵乘法 Loop
    // 注意:此处加入 OpenMP 多线程加速
    #pragma omp parallel for
    for(size_t i = 0; i < M; i++){
        for(size_t j = 0; j < N; j++){
            float sum = 0.0f;
            
            // 内积：Input的第i行 * Weight的第j行
            for(size_t index = 0; index < K; index++){
                float x_val = utils::cast<float>(in_ptr[index + i * K]);
                float y_val = utils::cast<float>(weight_ptr[index + j * K]);
                sum += x_val * y_val;
            }
            
            if(bias_ptr) sum += utils::cast<float>(bias_ptr[j]);
            out_ptr[i * N + j] = utils::cast<T>(sum);
            
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