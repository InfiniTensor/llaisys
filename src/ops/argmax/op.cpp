#include "op.hpp"
#include "../../utils.hpp"
namespace llaisys::ops {
    template<typename T>
    void argmax_cpu_kernel(tensor_t max_idx,tensor_t max_val,const tensor_t vals){
        const T*src=reinterpret_cast<const T*>(vals->data());
        T*dst_val=reinterpret_cast<T*>(max_val->data());
        int64_t*dst_idx=reinterpret_cast<int64_t*>(max_idx->data());
        
        // 获取最后一维的大小（vocab_size）
        size_t last_dim = vals->shape().back();
        size_t num_rows = vals->numel() / last_dim;  // 其他维度的乘积
        
        // 对每一行分别求 argmax
        for(size_t row = 0; row < num_rows; row++){
            float cur_max_fval = utils::cast<float>(src[row * last_dim]);
            T cur_max_val = src[row * last_dim];
            size_t cur_max_idx = 0;
            
            for(size_t i = 1; i < last_dim; i++){
                float cur_fval = utils::cast<float>(src[row * last_dim + i]);
                if(cur_fval > cur_max_fval){
                    cur_max_fval = cur_fval;
                    cur_max_idx = i;
                    cur_max_val = src[row * last_dim + i];
                }
            }
            dst_val[row] = cur_max_val;
            dst_idx[row] = cur_max_idx;
        }
    }
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    switch (vals->dtype()) {
        case LLAISYS_DTYPE_F16:
            argmax_cpu_kernel<llaisys::fp16_t>(max_idx, max_val, vals);
            break;
        case LLAISYS_DTYPE_BF16:
            argmax_cpu_kernel<llaisys::bf16_t>(max_idx, max_val, vals);
            break;
        case LLAISYS_DTYPE_F32:
            argmax_cpu_kernel<float>(max_idx, max_val, vals);
            break;
        default:
            throw std::runtime_error("Not support this dtype!");
    }
}
} // namespace llaisys::ops
