#include "llaisys/models/qwen2.h"
#include "llaisys/ops.h"
#include "llaisys_tensor.hpp"

#include <cstring>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <memory>


__C {
    struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
        // Validate input parameters
        if (!meta || !device_ids || ndevice <= 0) {
            std::cerr << "Invalid parameters for Qwen2 model creation" << std::endl;
            return nullptr;
        }

        // Allocate main model structure
        auto model = static_cast<LlaisysQwen2Model*>(std::calloc(1, sizeof(LlaisysQwen2Model)));
        if (!model) {
            std::cerr << "Failed to allocate LlaisysQwen2Model" << std::endl;
            return nullptr;
        }

        // Initialize metadata
        model->meta = static_cast<LlaisysQwen2Meta*>(std::malloc(sizeof(LlaisysQwen2Meta)));
        if (!model->meta) {
            std::cerr << "Failed to allocate model metadata" << std::endl;
            free(model);
            return nullptr;
        }
        std::memcpy(model->meta, meta, sizeof(LlaisysQwen2Meta));

        // Initialize weights structure  
        model->weights = static_cast<LlaisysQwen2Weights*>(std::calloc(1, sizeof(LlaisysQwen2Weights)));
        if (!model->weights) {
            std::cerr << "Failed to allocate model weights" << std::endl;
            free(model->meta);
            free(model);
            return nullptr;
        }

        // Initialize device information
        model->device = device;
        model->ndevice = ndevice;
        model->device_ids = static_cast<int*>(std::malloc(sizeof(int) * ndevice));
        if (!model->device_ids) {
            std::cerr << "Failed to allocate device IDs" << std::endl;
            free(model->weights);
            free(model->meta);
            free(model);
            return nullptr;
        }
        std::memcpy(model->device_ids, device_ids, sizeof(int) * ndevice);



        // Input Embedding
        size_t shape_in_embed[2] = { meta->voc, meta->hs };
        model->weights->in_embed = tensorCreate(shape_in_embed, 2, meta->dtype, device, device_ids[0]);

        // Helper functions for tensor array allocation
        auto alloc_layer_array_2d = [&](llaisysTensor_t *&ptr, size_t dim0, size_t dim1) -> bool {
            ptr = static_cast<llaisysTensor_t*>(std::malloc(sizeof(llaisysTensor_t) * meta->nlayer));
            if (!ptr) return false;
            
            for (size_t i = 0; i < meta->nlayer; ++i) {
                size_t shape[2] = { dim0, dim1 };
                ptr[i] = tensorCreate(shape, 2, meta->dtype, model->device, model->device_ids[0]);
                if (!ptr[i]) {
                    // Clean up previously allocated tensors on failure
                    for (size_t j = 0; j < i; ++j) {
                        tensorDestroy(ptr[j]);
                    }
                    free(ptr);
                    ptr = nullptr;
                    return false;
                }
            }
            return true;
        };

        auto alloc_layer_array_1d = [&](llaisysTensor_t *&ptr, size_t dim0) -> bool {
            ptr = static_cast<llaisysTensor_t*>(std::malloc(sizeof(llaisysTensor_t) * meta->nlayer));
            if (!ptr) return false;
            
            for (size_t i = 0; i < meta->nlayer; ++i) {
                size_t shape[1] = { dim0 };
                ptr[i] = tensorCreate(shape, 1, meta->dtype, model->device, model->device_ids[0]);
                if (!ptr[i]) {
                    // Clean up previously allocated tensors on failure
                    for (size_t j = 0; j < i; ++j) {
                        tensorDestroy(ptr[j]);
                    }
                    free(ptr);
                    ptr = nullptr;
                    return false;
                }
            }
            return true;
        };

        // Self-Attention
        alloc_layer_array_1d(model->weights->attn_norm_w, meta->hs);                     // [1536]
        alloc_layer_array_2d(model->weights->attn_q_w, meta->hs, meta->nh * meta->dh);   // [1536, 1536]
        alloc_layer_array_1d(model->weights->attn_q_b, meta->nh * meta->dh);             // [1536]
        alloc_layer_array_2d(model->weights->attn_k_w, meta->nkvh * meta->dh, meta->hs); // [256, 1536]
        alloc_layer_array_1d(model->weights->attn_k_b, meta->nkvh * meta->dh);           // [256]
        alloc_layer_array_2d(model->weights->attn_v_w, meta->nkvh * meta->dh, meta->hs); // [256, 1536]
        alloc_layer_array_1d(model->weights->attn_v_b, meta->nkvh * meta->dh);           // [256]
        alloc_layer_array_2d(model->weights->attn_o_w, meta->nh * meta->dh, meta->hs);   // [1536, 1536]

        // MLP
        alloc_layer_array_1d(model->weights->mlp_norm_w, meta->hs);             // [1536]
        alloc_layer_array_2d(model->weights->mlp_gate_w, meta->di, meta->hs);   // [8960, 1536]
        alloc_layer_array_2d(model->weights->mlp_up_w, meta->di, meta->hs);     // [8960, 1536]
        alloc_layer_array_2d(model->weights->mlp_down_w, meta->hs, meta->di);   // [1536, 8960]

        // Output Layer Norm
        size_t shape_out_norm[1] = { meta->hs };
        model->weights->out_norm_w = tensorCreate(shape_out_norm, 1, meta->dtype, model->device, model->device_ids[0]);

        // Output Embedding
        size_t shape_out_embed[2] = { meta->voc, meta->hs };
        model->weights->out_embed = tensorCreate(shape_out_embed, 2, meta->dtype, device, device_ids[0]);

        return model;
    }

    void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model) {
        if (!model) return;

        tensorDestroy(model->weights->in_embed);
        tensorDestroy(model->weights->out_embed);
        tensorDestroy(model->weights->out_norm_w);

        // attn_norm_w (1d tensor array)
        if (model->weights->attn_norm_w) {
            for (size_t i = 0; i < model->meta->nlayer; ++i) {
                tensorDestroy(model->weights->attn_norm_w[i]);
            }
            free(model->weights->attn_norm_w);
        }

        // attn_q_w (2d tensor array)
        if (model->weights->attn_q_w) {
            for (size_t i = 0; i < model->meta->nlayer; ++i) {
                tensorDestroy(model->weights->attn_q_w[i]);
            }
            free(model->weights->attn_q_w);
        }

        // attn_q_b (1d tensor array)
        if (model->weights->attn_q_b) {
            for (size_t i = 0; i < model->meta->nlayer; ++i) {
                tensorDestroy(model->weights->attn_q_b[i]);
            }
            free(model->weights->attn_q_b);
        }

        // attn_k_w (2d tensor array)
        if (model->weights->attn_k_w) {
            for (size_t i = 0; i < model->meta->nlayer; ++i) {
                tensorDestroy(model->weights->attn_k_w[i]);
            }
            free(model->weights->attn_k_w);
        }

        // attn_k_b (1d tensor array)
        if (model->weights->attn_k_b) {
            for (size_t i = 0; i < model->meta->nlayer; ++i) {
                tensorDestroy(model->weights->attn_k_b[i]);
            }
            free(model->weights->attn_k_b);
        }

        // attn_v_w (2d tensor array)
        if (model->weights->attn_v_w) {
            for (size_t i = 0; i < model->meta->nlayer; ++i) {
                tensorDestroy(model->weights->attn_v_w[i]);
            }
            free(model->weights->attn_v_w);
        }

        // attn_v_b (1d tensor array)
        if (model->weights->attn_v_b) {
            for (size_t i = 0; i < model->meta->nlayer; ++i) {
                tensorDestroy(model->weights->attn_v_b[i]);
            }
            free(model->weights->attn_v_b);
        }

        // attn_o_w (2d tensor array)
        if (model->weights->attn_o_w) {
            for (size_t i = 0; i < model->meta->nlayer; ++i) {
                tensorDestroy(model->weights->attn_o_w[i]);
            }
            free(model->weights->attn_o_w);
        }

        // mlp_norm_w (1d tensor array)
        if (model->weights->mlp_norm_w) {
            for (size_t i = 0; i < model->meta->nlayer; ++i) {
                tensorDestroy(model->weights->mlp_norm_w[i]);
            }
            free(model->weights->mlp_norm_w);
        }

        // mlp_gate_w (2d tensor array)
        if (model->weights->mlp_gate_w) {
            for (size_t i = 0; i < model->meta->nlayer; ++i) {
                tensorDestroy(model->weights->mlp_gate_w[i]);
            }
            free(model->weights->mlp_gate_w);
        }

        // mlp_up_w (2d tensor array)
        if (model->weights->mlp_up_w) {
            for (size_t i = 0; i < model->meta->nlayer; ++i) {
                tensorDestroy(model->weights->mlp_up_w[i]);
            }
            free(model->weights->mlp_up_w);
        }

        // mlp_down_w (2d tensor array)
        if (model->weights->mlp_down_w) {
            for (size_t i = 0; i < model->meta->nlayer; ++i) {
                tensorDestroy(model->weights->mlp_down_w[i]);
            }
            free(model->weights->mlp_down_w);
        }

        if (model->device_ids) {
            free(model->device_ids);
        }

        if (model->meta) {
            free(model->meta);
        }

        free(model);
    }

    struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model) {
        return model->weights;
    }

    int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken,
                                   llaisysTensor_t *kcache, llaisysTensor_t *vcache, size_t past_len,
                                   float temperature, int top_k, float top_p, int64_t seed) {
        if (!model || !token_ids || ntoken == 0) return -1;

        // If kv_cache != nullptr, it means KV Cache is used for performance.
        bool kv_cache_used = (kcache != nullptr && vcache != nullptr);
        // kcache [max_seq, nkvh, d], vcache [max_seq, nkvh, dv]

        size_t seqlen = ntoken;
        size_t hs = model->meta->hs;       // hidden size
        size_t nh = model->meta->nh;       // num heads
        size_t dh = model->meta->dh;       // head dim
        size_t nkvh = model->meta->nkvh;   // num key-value heads
        size_t di = model->meta->di;       // mlp intermediate dim
        size_t voc = model->meta->voc;     // vocab size
        size_t nlayer = model->meta->nlayer;
        float scale = 1.0f / std::sqrt(static_cast<float>(dh)); // scale for attention
        float rms_eps = model->meta->epsilon; // epsilon for RMSNorm
        float rope_theta = model->meta->theta; // theta for RoPE
        
        // 1. intput token_ids -> tensor
        size_t input_tensor_shape[1] = {seqlen};
        
        llaisysTensor_t input_tensor = tensorCreate(input_tensor_shape, 1, LLAISYS_DTYPE_I64, model->device, model->device_ids[0]);
        
        tensorLoad(input_tensor, token_ids);
        
        // 2. Embedding lookup: [seqlen] -> [seqlen, hs]
        size_t output_embedding_tensor_shape[2] = {seqlen, hs};
        llaisysTensor_t output_embedding_tensor = tensorCreate(output_embedding_tensor_shape, 2, model->meta->dtype, model->device, model->device_ids[0]);
        llaisysEmbedding(output_embedding_tensor, input_tensor, model->weights->in_embed);
        
        
        
        // output_hidden_layer_tensor is used to store the output of the hidden layer
        llaisysTensor_t output_hidden_layer_tensor = output_embedding_tensor;
        size_t position_shape[1] = {seqlen};
        llaisysTensor_t position_ids = tensorCreate(position_shape, 1, LLAISYS_DTYPE_I64, model->device, model->device_ids[0]);
        int64_t* pos_data_cpu = new int64_t[seqlen];
        for (size_t i = 0; i < seqlen; i++) {
            if(kv_cache_used)
                pos_data_cpu[i] = (int64_t)(past_len + i);  // When using KV cache, position ids continue from past_len
            else
                pos_data_cpu[i] = (int64_t) i;
        }
        tensorLoad(position_ids, pos_data_cpu);
        delete[] pos_data_cpu;
        
        

        // 3. Transformer hidden layers
        for (size_t i = 0; i < nlayer; i++) {
            // 3.1 LayerNorm before Self-attention
            llaisysTensor_t attn_norm_w = model->weights->attn_norm_w[i];
            size_t output_input_layernorm_shape[2] = {seqlen, hs};
            llaisysTensor_t output_input_layernorm_tensor = tensorCreate(output_input_layernorm_shape, 2, model->meta->dtype, model->device, model->device_ids[0]);
            llaisysRmsNorm(output_input_layernorm_tensor, output_hidden_layer_tensor, attn_norm_w, rms_eps);



            // 3.2 Q projection
            llaisysTensor_t attn_q_w = model->weights->attn_q_w[i];
            llaisysTensor_t attn_q_b = model->weights->attn_q_b[i];
            size_t q_tensor_shape[2] = {seqlen, nh * dh};
            llaisysTensor_t q_tensor = tensorCreate(q_tensor_shape, 2, model->meta->dtype, model->device, model->device_ids[0]);
            llaisysLinear(q_tensor, output_input_layernorm_tensor, attn_q_w, attn_q_b);

            size_t q_tensor_reshape_shape[3] = {seqlen, nh, dh};
            q_tensor = tensorReshape(q_tensor, q_tensor_reshape_shape, 3);
            llaisysTensor_t q_rope_tensor = tensorCreate(q_tensor_reshape_shape, 3, model->meta->dtype, model->device, model->device_ids[0]);
            llaisysROPE(q_rope_tensor, q_tensor, position_ids, rope_theta);



            // 3.3 K projection
            llaisysTensor_t attn_k_w = model->weights->attn_k_w[i];
            llaisysTensor_t attn_k_b = model->weights->attn_k_b[i];
            size_t k_tensor_shape[2] = {seqlen, nkvh * dh};
            llaisysTensor_t k_tensor = tensorCreate(k_tensor_shape, 2, model->meta->dtype, model->device, model->device_ids[0]);
            llaisysLinear(k_tensor, output_input_layernorm_tensor, attn_k_w, attn_k_b);

            size_t k_tensor_reshape_shape[3] = {seqlen, nkvh, dh};
            k_tensor = tensorReshape(k_tensor, k_tensor_reshape_shape, 3);
            llaisysTensor_t k_rope_tensor; 
            if (kv_cache_used) {
                // When using KV cache, we need to update the kcache tensor
                // kcache shape: [max_seq, nkvh, dh]
                // Slice the kcache to get the current position to update
                k_rope_tensor = tensorSlice(kcache[i], 0, past_len, past_len + seqlen); // Write to kcache
            } else {
                k_rope_tensor = tensorCreate(k_tensor_reshape_shape, 3, model->meta->dtype, model->device, model->device_ids[0]);
            }
            llaisysROPE(k_rope_tensor, k_tensor, position_ids, rope_theta);



            // 3.4 V projection
            llaisysTensor_t attn_v_w = model->weights->attn_v_w[i];
            llaisysTensor_t attn_v_b = model->weights->attn_v_b[i];
            size_t v_tensor_shape[2] = {seqlen, nkvh * dh};
            llaisysTensor_t v_tensor;
            if (kv_cache_used) {
                // When using KV cache, we need to update the vcache tensor
                // vcache shape: [max_seq, nkvh, dh]
                // Slice the vcache to get the current position to update
                v_tensor = tensorView(tensorSlice(vcache[i], 0, past_len, past_len + seqlen), v_tensor_shape, 2);
            } else {
                v_tensor = tensorCreate(v_tensor_shape, 2, model->meta->dtype, model->device, model->device_ids[0]);
            }
            llaisysLinear(v_tensor, output_input_layernorm_tensor, attn_v_w, attn_v_b);


            // 3.5 Self-attention with Flash Attention
            size_t output_self_attn_multihead_tensor_shape[3] = {seqlen, nh, dh};
            llaisysTensor_t output_self_attn_tensor = tensorCreate(output_self_attn_multihead_tensor_shape, 3, model->meta->dtype, model->device, model->device_ids[0]);
            
            // Prepare tensors for attention (with or without KV cache)
            llaisysTensor_t k_for_attn, v_for_attn;
            size_t kv_seq_len;
            
            if (kv_cache_used) {
                // Use KV cache to speed up inference
                // kcache [max_seq, nkvh, d], vcache [max_seq, nkvh, dv]
                kv_seq_len = past_len + seqlen;
                k_for_attn = tensorSlice(kcache[i], 0, 0, kv_seq_len);
                v_for_attn = tensorSlice(vcache[i], 0, 0, kv_seq_len);
            } else {
                size_t v_tensor_reshape_shape[3] = {seqlen, nkvh, dh};
                v_for_attn = tensorReshape(v_tensor, v_tensor_reshape_shape, 3);
                k_for_attn = k_rope_tensor;
                kv_seq_len = seqlen;
            }
           
            
            llaisysSelfAttention(output_self_attn_tensor, q_rope_tensor, k_for_attn, v_for_attn, scale);
            
            size_t output_self_attn_tensor_shape[2] = {seqlen, nh * dh};
            output_self_attn_tensor = tensorReshape(output_self_attn_tensor, output_self_attn_tensor_shape, 2);
            
            
            // 3.6 Self-attention output projection
            llaisysTensor_t attn_o_w = model->weights->attn_o_w[i];
            size_t o_tensor_shape[2] = {seqlen, hs};
            llaisysTensor_t o_tensor = tensorCreate(o_tensor_shape, 2, model->meta->dtype, model->device, model->device_ids[0]);
            
            llaisysLinear(o_tensor, output_self_attn_tensor, attn_o_w, nullptr);

            // 3.7 Residual connection after attn
            size_t output_res1_tensor_shape[2] = {seqlen, hs};
            llaisysTensor_t output_res1_tensor = tensorCreate(output_res1_tensor_shape, 2, model->meta->dtype, model->device, model->device_ids[0]);
            llaisysAdd(output_res1_tensor, output_hidden_layer_tensor, o_tensor);


            // 3.8 Post-attention LayerNorm
            llaisysTensor_t post_attn_norm_w = model->weights->mlp_norm_w[i];
            size_t output_post_self_attn_layernorm_tensor_shape[2] = {seqlen, hs};
            llaisysTensor_t output_post_self_attn_layernorm_tensor = tensorCreate(output_post_self_attn_layernorm_tensor_shape, 2, model->meta->dtype, model->device, model->device_ids[0]);
            llaisysRmsNorm(output_post_self_attn_layernorm_tensor, output_res1_tensor, post_attn_norm_w, rms_eps);


            // 3.9 MLP (Gate, Up, Down)
            llaisysTensor_t mlp_gate_w = model->weights->mlp_gate_w[i];
            llaisysTensor_t mlp_up_w = model->weights->mlp_up_w[i];
            llaisysTensor_t mlp_down_w = model->weights->mlp_down_w[i];

            size_t mlp_gate_tensor_shape[2] = {seqlen, di};
            llaisysTensor_t mlp_gate_tensor = tensorCreate(mlp_gate_tensor_shape, 2, model->meta->dtype, model->device, model->device_ids[0]);
            llaisysLinear(mlp_gate_tensor, output_post_self_attn_layernorm_tensor, mlp_gate_w, nullptr);


            size_t mlp_up_tensor_shape[2] = {seqlen, di};
            llaisysTensor_t mlp_up_tensor = tensorCreate(mlp_up_tensor_shape, 2, model->meta->dtype, model->device, model->device_ids[0]);
            llaisysLinear(mlp_up_tensor, output_post_self_attn_layernorm_tensor, mlp_up_w, nullptr);


            size_t swiglu_tensor_shape[2] = {seqlen, di};
            llaisysTensor_t swiglu_tensor = tensorCreate(swiglu_tensor_shape, 2, model->meta->dtype, model->device, model->device_ids[0]);
            llaisysSwiGLU(swiglu_tensor, mlp_gate_tensor, mlp_up_tensor);


            size_t mlp_down_tensor_shape[2] = {seqlen, hs};
            llaisysTensor_t mlp_down_tensor = tensorCreate(mlp_down_tensor_shape, 2, model->meta->dtype, model->device, model->device_ids[0]);
            llaisysLinear(mlp_down_tensor, swiglu_tensor, mlp_down_w, nullptr);



            // 3.10 Residual connection after MLP
            llaisysAdd(output_hidden_layer_tensor, output_res1_tensor, mlp_down_tensor);





            if(!kv_cache_used){
                tensorDestroy(k_rope_tensor);
                tensorDestroy(v_tensor);
            }
            
            // release intermediate tensors
            tensorDestroy(output_input_layernorm_tensor);
            tensorDestroy(q_tensor);
            tensorDestroy(q_rope_tensor);
            tensorDestroy(k_tensor);
            tensorDestroy(output_self_attn_tensor);
            tensorDestroy(o_tensor);
            tensorDestroy(output_res1_tensor);
            tensorDestroy(output_post_self_attn_layernorm_tensor);
            tensorDestroy(mlp_gate_tensor);
            tensorDestroy(mlp_up_tensor);
            tensorDestroy(swiglu_tensor);
            tensorDestroy(mlp_down_tensor);
        }
        // 4. Output LayerNorm
        llaisysTensor_t final_layernorm_w = model->weights->out_norm_w;
        size_t output_final_layernorm_shape[2] = {seqlen, hs};
        llaisysTensor_t output_final_layernorm_tensor = tensorCreate(output_final_layernorm_shape, 2, model->meta->dtype, model->device, model->device_ids[0]);
        llaisysRmsNorm(output_final_layernorm_tensor, output_hidden_layer_tensor, final_layernorm_w, rms_eps);



        // 5. Output [seqlen, voc]
        size_t output_tensor_shape[2] = {seqlen, voc};
        llaisysTensor_t output_tensor = tensorCreate(output_tensor_shape, 2, model->meta->dtype, model->device, model->device_ids[0]);
        llaisysLinear(output_tensor, output_final_layernorm_tensor, model->weights->out_embed, nullptr);
    


        // [seqlen, voc] -> [1, voc]
        size_t output_tensor_slice_reshape_shape[1] = {voc};
        llaisysTensor_t output_tensor_slice = tensorSlice(output_tensor, 0, seqlen-1, seqlen); // only need the last token's logits


        size_t index_shape[1] = {1};
        llaisysTensor_t index_tensor = tensorCreate(index_shape, 1, LLAISYS_DTYPE_I64, model->device, model->device_ids[0]);
        llaisysTensor_t value_tensor = tensorCreate(index_shape, 1, model->meta->dtype, model->device, model->device_ids[0]);
        llaisysRandomSample(index_tensor, value_tensor, tensorView(output_tensor_slice, output_tensor_slice_reshape_shape, 1),
                    temperature, top_k, top_p, seed);


        // For NVIDIA device, need to copy data back to CPU to read it
        int64_t index;
        if (model->device == LLAISYS_DEVICE_NVIDIA) {
            // Use C++ API to convert tensor to CPU
            auto index_cpu_tensor = index_tensor->tensor->to(LLAISYS_DEVICE_CPU);
            index = *((int64_t *) index_cpu_tensor->data());
        } else {
            index = *((int64_t *) tensorGetData(index_tensor));
        }

        tensorDestroy(position_ids);
        tensorDestroy(input_tensor);
        tensorDestroy(output_embedding_tensor);
        tensorDestroy(output_final_layernorm_tensor);
        tensorDestroy(output_tensor);
        tensorDestroy(index_tensor);
        tensorDestroy(value_tensor);

        return index;
    }
}   