#include "op.hpp"
#include <cmath>
#include <vector>
#include <algorithm>

const int INF = 0x3f3f3f3f;

namespace llaisys::ops {

template<typename T>
void self_attention_kernel(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale){
    T* attn_val_ptr_base = reinterpret_cast<T*>(attn_val->data());
    const T* q_ptr_base = reinterpret_cast<const T*>(q->data());
    const T* k_ptr_base = reinterpret_cast<const T*>(k->data());
    const T* v_ptr_base = reinterpret_cast<const T*>(v->data());

    size_t batch, seqlen, nhead, d, total_len, nkvhead, dv;
    
    // 【关键】智能判断模式
    // DeepSeek 的 Hidden Dim 是 1536 或 256
    size_t last_dim = q->shape().back();
    bool is_deepseek = (last_dim == 1536 || last_dim == 256);

    if (is_deepseek) {
        // --------------------------------------------------------
        // 模式 A: DeepSeek 推理模式 [Batch, Seq, Hidden]
        // 说明：
        //   - q 始终是 [B, T, H]，H=nh*head_dim (例如 1536 = 12 * 128)
        //   - k/v 在两种场景下：
        //       1) 即时计算时为 [B, T, kv_dim]，其中 kv_dim = nkvh * head_dim
        //       2) 从 KV Cache 读出后为 [B, T_total, nkvh, head_dim]
        // --------------------------------------------------------
        batch = q->shape()[0];
        seqlen = q->shape()[1];
        size_t hidden_q = q->shape()[2];

        // 拆 head：Q 的 head 维度始终按 128 处理
        d = 128;
        nhead = hidden_q / d; // 例如 1536 / 128 = 12

        total_len = k->shape()[1];

        if (k->shape().size() == 3) {
            // [B, T_total, kv_dim] 视为 [B, T_total, nkvh, head_dim] 的拍扁形式
            size_t hidden_kv = k->shape()[2];
            dv = d;
            nkvhead = hidden_kv / dv; // 例如 256 / 128 = 2
        } else if (k->shape().size() == 4) {
            // [B, T_total, nkvh, head_dim] —— 来自 KV Cache 的 4D 形式
            nkvhead = k->shape()[2];
            dv = k->shape()[3];       // 一般为 128
        } else {
            throw std::runtime_error("Unsupported K shape for DeepSeek mode");
        }
    } 
    else if (k->shape().size() == 4) {
        // --------------------------------------------------------
        // 模式 B: 4D 标准模式 [Batch, Seq, Head, Dim]
        // --------------------------------------------------------
        batch = q->shape()[0];
        seqlen = q->shape()[1];
        size_t hidden_size = q->shape()[2]; 
        total_len = k->shape()[1];
        nkvhead = k->shape()[2];
        d = k->shape()[3];
        nhead = hidden_size / d;
        dv = d;
    }
    else {
        // --------------------------------------------------------
        // 模式 C: 单元测试/通用兼容模式 [Seq, Head, Dim] (你最初的逻辑)
        // --------------------------------------------------------
        // 这里的 shape[0] 被视为 SeqLen，Batch 被视为 1
        batch = 1; 
        seqlen = q->shape()[0];
        nhead = q->shape()[1];
        d = q->shape()[2];

        total_len = k->shape()[0];
        nkvhead = k->shape()[1];
        dv = v->shape()[2];
    }

    // 计算 Stride (内存跨度)
    size_t stride_q = seqlen * nhead * d;
    size_t stride_k = total_len * nkvhead * d;
    size_t stride_v = total_len * nkvhead * dv;
    size_t stride_out = seqlen * nhead * dv;

    // 针对 DeepSeek 的广播检查
    if (is_deepseek) {
        size_t batch_k = k->shape()[0];
        size_t batch_v = v->shape()[0];
        if (batch_k < batch) stride_k = 0;
        if (batch_v < batch) stride_v = 0;
    }

    // 执行 Batch 循环
    // 注意：在模式 C (单元测试) 下，batch=1，只会跑一次，完美复现你最初的逻辑
    for (size_t b = 0; b < batch; b++) {
        
        T* attn_val_ptr = attn_val_ptr_base + b * stride_out;
        const T* q_ptr = q_ptr_base + b * stride_q;
        const T* k_ptr = k_ptr_base + b * stride_k;
        const T* v_ptr = v_ptr_base + b * stride_v;

        std::vector<T> A(total_len);
        size_t group_size = (nkvhead == 0) ? 1 : (nhead / nkvhead);
        if(group_size == 0) group_size = 1;
        
        auto get_k_index = [&](size_t n_index) -> size_t {
            if(nkvhead == 1) return 0;
            if(nkvhead == nhead) return n_index;
            return n_index / group_size; 
        };

        for(size_t n_index = 0; n_index < nhead; n_index++){
            size_t k_index = get_k_index(n_index);
            for(size_t i = 0; i < seqlen; i++){
                float MAX_num = -1e30f;
                // Q * K
                for(size_t j = 0; j < total_len; j++){
                    float sum = 0.0f;
                    for(size_t k = 0; k < d; k++){
                        float q_val = utils::cast<float>(q_ptr[i*(nhead*d)+n_index*d+k]);
                        float k_val = utils::cast<float>(k_ptr[j*(nkvhead*d)+k_index*d+k]);
                        sum += q_val * k_val;
                    }
                    sum *= scale;
                    // Causal Mask logic
                    size_t global_i = total_len - seqlen + i;
                    if(j > global_i){
                        A[j] = utils::cast<T>(-INF);
                    }
                    else{
                        A[j] = utils::cast<T>(sum);
                        MAX_num = std::fmax(MAX_num, sum);
                    }
                }
                
                // Softmax
                float softmax_accu = 0;
                for(size_t j = 0; j < total_len; j++){
                    float a_val = utils::cast<float>(A[j]);
                    softmax_accu += std::exp(a_val - MAX_num);
                }
                
                float inv_accu = 1.0f / softmax_accu;
                for(size_t j = 0; j < total_len; j++){
                    float a_val = utils::cast<float>(A[j]);
                    float a_sval = std::exp(a_val - MAX_num);
                    A[j] = utils::cast<T>(a_sval * inv_accu);
                }
                
                // Weighted Sum
                for(size_t j = 0; j < dv; j++){
                    float out_sum = 0.0f;
                    for(size_t k = 0; k < total_len; k++){
                        float a_cv = utils::cast<float>(A[k]);
                        float v_cv = utils::cast<float>(v_ptr[k*nkvhead*dv+k_index*dv+j]);
                        out_sum += a_cv * v_cv;
                    }
                    attn_val_ptr[i*nhead*dv+n_index*dv+j] = utils::cast<T>(out_sum);
                }
            }
        }
    }
}

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    tensor_t cq = q->isContiguous() ? q : q->contiguous();
    tensor_t ck = k->isContiguous() ? k : k->contiguous();
    tensor_t cv = v->isContiguous() ? v : v->contiguous();
    switch (q->dtype()) {
        case LLAISYS_DTYPE_F16:
            self_attention_kernel<llaisys::fp16_t>(attn_val,cq,ck,cv,scale);
            break;
        case LLAISYS_DTYPE_BF16:
            self_attention_kernel<llaisys::bf16_t>(attn_val,cq,ck,cv,scale);
            break;
        case LLAISYS_DTYPE_F32:
            self_attention_kernel<float>(attn_val,cq,ck,cv,scale);
            break;
        default:
            throw std::runtime_error("Not support this dtype!");
    }
}
} // namespace llaisys::ops