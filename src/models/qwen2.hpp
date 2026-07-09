#pragma once

#include "llaisys/models/qwen2.h"

#include "../llaisys/llaisys_tensor.hpp"
#include "../tensor/tensor.hpp"

#include <vector>

namespace llaisys::models {

class Qwen2Model {
public:
    Qwen2Model(const LlaisysQwen2Meta &meta, llaisysDeviceType_t device, int *device_ids, int ndevice);
    ~Qwen2Model();

    LlaisysQwen2Weights *weights();
    int64_t infer(const int64_t *token_ids, size_t ntoken, float temperature, int top_k, float top_p);

private:
    llaisysTensor_t create_weight_tensor(const std::vector<size_t> &shape);
    llaisys::tensor_t unwrap(llaisysTensor_t tensor) const;
    void zero_tensor(llaisysTensor_t tensor);
    void zero_tensor_data(const llaisys::tensor_t &tensor);
    void write_kv_cache(size_t layer, size_t pos, const llaisys::tensor_t &k, const llaisys::tensor_t &v);
    void reset_state();

    llaisys::tensor_t forward_token(int64_t token_id, size_t pos);

private:
    LlaisysQwen2Meta _meta;
    llaisysDeviceType_t _device;
    int _device_id;
    LlaisysQwen2Weights _weights;
    std::vector<llaisysTensor_t> _owned_tensors;
    std::vector<llaisys::tensor_t> _k_cache;
    std::vector<llaisys::tensor_t> _v_cache;
    size_t _past_len;
    llaisys::tensor_t _zero_bias_hs;
    llaisys::tensor_t _zero_bias_di;
    llaisys::tensor_t _zero_bias_voc;
};

} // namespace llaisys::models
