#include "op.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    //TO_BE_IMPLEMENTED();

    // --- 基本检查 ---
    if (!out || !in || !weight) {
        throw std::runtime_error("linear: out/in/weight must not be null");
    }

    auto out_shape = out->shape();
    auto in_shape = in->shape();
    auto w_shape   = weight->shape();

    if (in_shape.size() != 2 || w_shape.size() != 2 || out_shape.size() != 2) {
        throw std::runtime_error("linear: only supports 2D tensors (batch, features)");
    }

    const size_t B  = in_shape[0];
    const size_t K  = in_shape[1];      // in_features
    const size_t OC = w_shape[0];       // out_features  ✅ 正确
    const size_t WK = w_shape[1];       // should equal K

    if (WK != K) {
        throw std::runtime_error("linear: weight shape mismatch, expected w.shape[1] == in.shape[1]");
    }
    if (out_shape[0] != B || out_shape[1] != OC) {
        throw std::runtime_error("linear: out shape mismatch, expected out.shape == (B, out_features)");
    }

    const bool has_bias = (bias != nullptr);
    if (has_bias) {
        auto b_shape = bias->shape();
        if (b_shape.size() != 1 || b_shape[0] != OC) {
            throw std::runtime_error("linear: bias shape mismatch, expected bias.shape == (out_features,)");
        }
    }

    // dtype 建议也校验一致（至少 in/weight/out 一致）
    auto dtype = in->dtype();
    if (weight->dtype() != dtype || out->dtype() != dtype) {
        throw std::runtime_error("linear: dtype mismatch among in/weight/out");
    }
    if (has_bias && bias->dtype() != dtype) {
        throw std::runtime_error("linear: bias dtype mismatch");
    }



    
    // // 检查 bias 是否提供
    // bool has_bias = (bias != nullptr);
    // const std::byte* bias_data = has_bias ? bias->data() : nullptr;
    
    // // 获取维度信息
    // auto out_shape = out->shape();
    // auto in_shape = in->shape();
    // auto weight_shape = weight->shape();
    
    // size_t batch_size = in_shape[0];
    // size_t in_features = in_shape[1];
    // size_t out_features = weight_shape[1];
    // auto dtype = in->dtype();

    auto in_data     = in->data();
    auto w_data      = weight->data();
    auto out_data    = out->data();
    const std::byte* b_data = has_bias ? bias->data() : nullptr;
    
    // 根据数据类型进行处理
    switch (static_cast<llaisysDataType_t>(dtype)) {
    case LLAISYS_DTYPE_F32: {
        // auto* in_ptr = reinterpret_cast<const float*>(in_data);
        // auto* weight_ptr = reinterpret_cast<const float*>(weight_data);
        // auto* out_ptr = reinterpret_cast<float*>(out_data);
        // const float* bias_ptr = has_bias ? reinterpret_cast<const float*>(bias_data) : nullptr;
        const float* x = reinterpret_cast<const float*>(in_data);
        const float* w = reinterpret_cast<const float*>(w_data);
        float* y       = reinterpret_cast<float*>(out_data);
        const float* b = has_bias ? reinterpret_cast<const float*>(b_data) : nullptr;
        
        // for (size_t b = 0; b < batch_size; b++) {
        //     const float* batch_in = in_ptr + b * in_features;
        //     float* batch_out = out_ptr + b * out_features;
        for (size_t n = 0; n < B; ++n) {
            const float* x_row = x + n * K;
            float* y_row       = y + n * OC;
            
            // // 初始化输出（如果有bias则加bias，否则为0）
            // if (has_bias) {
            //     for (size_t o = 0; o < out_features; o++) {
            //         batch_out[o] = bias_ptr[o];
            //     }
            // } else {
            //     for (size_t o = 0; o < out_features; o++) {
            //         batch_out[o] = 0.0f;
            //     }
            // }
            
            // // 矩阵乘法: out = in * weight^T
            // for (size_t i = 0; i < in_features; i++) {
            //     float in_val = batch_in[i];
            //     for (size_t o = 0; o < out_features; o++) {
            //         // weight 是 [out_features, in_features]
            //         // 需要 weight[o, i] 对应 weight^T[i, o]
            //         float weight_val = weight_ptr[o * in_features + i];
            //         batch_out[o] += in_val * weight_val;
            //     }
            // }
            for (size_t o = 0; o < OC; ++o) {
                float acc = has_bias ? b[o] : 0.0f;
                const float* w_row = w + o * K;  // w[o, :]
                for (size_t i = 0; i < K; ++i) {
                    acc += x_row[i] * w_row[i];
                }
                y_row[o] = acc;
            }

        }
        break;
    }
    
    case LLAISYS_DTYPE_F16: {
        // auto* in_ptr = reinterpret_cast<const llaisys::fp16_t*>(in_data);
        // auto* weight_ptr = reinterpret_cast<const llaisys::fp16_t*>(weight_data);
        // auto* out_ptr = reinterpret_cast<llaisys::fp16_t*>(out_data);
        // const llaisys::fp16_t* bias_ptr = has_bias ? reinterpret_cast<const llaisys::fp16_t*>(bias_data) : nullptr;
        const llaisys::fp16_t* x = reinterpret_cast<const llaisys::fp16_t*>(in_data);
        const llaisys::fp16_t* w = reinterpret_cast<const llaisys::fp16_t*>(w_data);
        llaisys::fp16_t* y       = reinterpret_cast<llaisys::fp16_t*>(out_data);
        const llaisys::fp16_t* b = has_bias ? reinterpret_cast<const llaisys::fp16_t*>(b_data) : nullptr;
        
        // for (size_t b = 0; b < batch_size; b++) {
        //     const llaisys::fp16_t* batch_in = in_ptr + b * in_features;
        //     llaisys::fp16_t* batch_out = out_ptr + b * out_features;
        for (size_t n = 0; n < B; ++n) {
            const llaisys::fp16_t* x_row = x + n * K;
            llaisys::fp16_t* y_row       = y + n * OC;
            
            // // 初始化输出
            // if (has_bias) {
            //     for (size_t o = 0; o < out_features; o++) {
            //         batch_out[o] = bias_ptr[o];
            //     }
            // } else {
            //     for (size_t o = 0; o < out_features; o++) {
            //         batch_out[o] = llaisys::utils::_f32_to_f16(0.0f);  // 对于 fp16_t
            //     }
            // }
            
            // // 矩阵乘法
            // for (size_t i = 0; i < in_features; i++) {
            //     float in_val = llaisys::utils::_f16_to_f32(batch_in[i]);
            //     for (size_t o = 0; o < out_features; o++) {
            //         float weight_val = llaisys::utils::_f16_to_f32(weight_ptr[o * in_features + i]);
            //         float out_val = llaisys::utils::_f16_to_f32(batch_out[o]);
            //         out_val += in_val * weight_val;
            //         batch_out[o] = llaisys::utils::_f32_to_f16(out_val);
            //     }
            // }

            for (size_t o = 0; o < OC; ++o) {
                float acc = has_bias ? llaisys::utils::_f16_to_f32(b[o]) : 0.0f;
                const llaisys::fp16_t* w_row = w + o * K;
                for (size_t i = 0; i < K; ++i) {
                    float xv = llaisys::utils::_f16_to_f32(x_row[i]);
                    float wv = llaisys::utils::_f16_to_f32(w_row[i]);
                    acc += xv * wv;
                }
                y_row[o] = llaisys::utils::_f32_to_f16(acc);
            }
        }
        break;
    }
    
    case LLAISYS_DTYPE_BF16: {
        // auto* in_ptr = reinterpret_cast<const llaisys::bf16_t*>(in_data);
        // auto* weight_ptr = reinterpret_cast<const llaisys::bf16_t*>(weight_data);
        // auto* out_ptr = reinterpret_cast<llaisys::bf16_t*>(out_data);
        // const llaisys::bf16_t* bias_ptr = has_bias ? reinterpret_cast<const llaisys::bf16_t*>(bias_data) : nullptr;
        const llaisys::bf16_t* x = reinterpret_cast<const llaisys::bf16_t*>(in_data);
        const llaisys::bf16_t* w = reinterpret_cast<const llaisys::bf16_t*>(w_data);
        llaisys::bf16_t* y       = reinterpret_cast<llaisys::bf16_t*>(out_data);
        const llaisys::bf16_t* b = has_bias ? reinterpret_cast<const llaisys::bf16_t*>(b_data) : nullptr;



        // for (size_t b = 0; b < batch_size; b++) {
        //     const llaisys::bf16_t* batch_in = in_ptr + b * in_features;
        //     llaisys::bf16_t* batch_out = out_ptr + b * out_features;

        for (size_t n = 0; n < B; ++n) {
            const llaisys::bf16_t* x_row = x + n * K;
            llaisys::bf16_t* y_row       = y + n * OC;
            
        //     // 初始化输出
        //     if (has_bias) {
        //         for (size_t o = 0; o < out_features; o++) {
        //             batch_out[o] = bias_ptr[o];
        //         }
        //     } else {
        //         for (size_t o = 0; o < out_features; o++) {
        //             //batch_out[o] = llaisys::bf16_t(0.0f);
        //             batch_out[o] = llaisys::utils::_f32_to_bf16(0.0f);
        //         }
        //     }
            
        //     // 矩阵乘法
        //     for (size_t i = 0; i < in_features; i++) {
        //         float in_val = llaisys::utils::_bf16_to_f32(batch_in[i]);
        //         for (size_t o = 0; o < out_features; o++) {
        //             float weight_val = llaisys::utils::_bf16_to_f32(weight_ptr[o * in_features + i]);
        //             float out_val = llaisys::utils::_bf16_to_f32(batch_out[o]);
        //             out_val += in_val * weight_val;
        //             batch_out[o] = llaisys::utils::_f32_to_bf16(out_val);
        //         }
        //     }

            for (size_t o = 0; o < OC; ++o) {
                float acc = has_bias ? llaisys::utils::_bf16_to_f32(b[o]) : 0.0f;
                const llaisys::bf16_t* w_row = w + o * K;
                for (size_t i = 0; i < K; ++i) {
                    float xv = llaisys::utils::_bf16_to_f32(x_row[i]);
                    float wv = llaisys::utils::_bf16_to_f32(w_row[i]);
                    acc += xv * wv;
                }
                y_row[o] = llaisys::utils::_f32_to_bf16(acc);
            }


        }
        break;
    }
    
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(static_cast<llaisysDataType_t>(dtype));
    }
}
} // namespace llaisys::ops
