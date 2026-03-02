#include "model.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "../../ops/add/op.hpp"
#include "../../device/runtime_api.hpp"
#include <cstring>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <limits>
#include <vector>

namespace llaisys::models::qwen2 {
namespace {
int64_t argmax_host(const std::vector<float> &vals) {
    ASSERT(!vals.empty(), "argmax_host: input must not be empty");
    size_t best = 0;
    for (size_t i = 1; i < vals.size(); ++i) {
        if (vals[i] > vals[best]) {
            best = i;
        }
    }
    return static_cast<int64_t>(best);
}

std::vector<float> logits_to_host_f32(tensor_t logits, const LlaisysRuntimeAPI *api) {
    const size_t n = logits->numel();
    std::vector<float> out(n);
    switch (logits->dtype()) {
    case LLAISYS_DTYPE_F32: {
        api->memcpy_sync(out.data(), logits->data(), n * sizeof(float), LLAISYS_MEMCPY_D2H);
        break;
    }
    case LLAISYS_DTYPE_F16: {
        std::vector<llaisys::fp16_t> tmp(n);
        api->memcpy_sync(tmp.data(), logits->data(), n * sizeof(llaisys::fp16_t), LLAISYS_MEMCPY_D2H);
        for (size_t i = 0; i < n; ++i) {
            out[i] = llaisys::utils::cast<float>(tmp[i]);
        }
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        std::vector<llaisys::bf16_t> tmp(n);
        api->memcpy_sync(tmp.data(), logits->data(), n * sizeof(llaisys::bf16_t), LLAISYS_MEMCPY_D2H);
        for (size_t i = 0; i < n; ++i) {
            out[i] = llaisys::utils::cast<float>(tmp[i]);
        }
        break;
    }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(logits->dtype());
    }
    return out;
}

int64_t sample_from_logits(
    const std::vector<float> &logits,
    int top_k,
    float top_p,
    float temperature) {
    ASSERT(!logits.empty(), "sample_from_logits: logits must not be empty");

    if (temperature <= 0.0f) {
        return argmax_host(logits);
    }

    const size_t vocab = logits.size();
    if (top_k <= 0 || top_k > static_cast<int>(vocab)) {
        top_k = static_cast<int>(vocab);
    }
    if (top_p <= 0.0f || top_p > 1.0f) {
        top_p = 1.0f;
    }

    if (top_k == 1 && top_p >= 1.0f) {
        return argmax_host(logits);
    }

    std::vector<int> idx(vocab);
    std::iota(idx.begin(), idx.end(), 0);
    auto by_logit_desc = [&logits](int a, int b) { return logits[a] > logits[b]; };
    if (top_k < static_cast<int>(vocab)) {
        std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(), by_logit_desc);
        idx.resize(top_k);
    }
    std::sort(idx.begin(), idx.end(), by_logit_desc);

    const float inv_temp = 1.0f / temperature;
    float max_scaled = -std::numeric_limits<float>::infinity();
    for (int i : idx) {
        max_scaled = std::max(max_scaled, logits[i] * inv_temp);
    }

    std::vector<double> probs(idx.size(), 0.0);
    double total = 0.0;
    for (size_t i = 0; i < idx.size(); ++i) {
        double p = std::exp(static_cast<double>(logits[idx[i]] * inv_temp - max_scaled));
        if (!std::isfinite(p) || p < 0.0) {
            p = 0.0;
        }
        probs[i] = p;
        total += p;
    }
    if (total <= 0.0) {
        return static_cast<int64_t>(idx.front());
    }

    if (top_p < 1.0f) {
        double cum = 0.0;
        size_t keep = 0;
        for (size_t i = 0; i < probs.size(); ++i) {
            cum += probs[i] / total;
            keep = i + 1;
            if (cum >= static_cast<double>(top_p)) {
                break;
            }
        }
        keep = std::max<size_t>(keep, 1);
        idx.resize(keep);
        probs.resize(keep);
    }

    thread_local std::mt19937 rng(std::random_device{}());
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    int chosen = dist(rng);
    return static_cast<int64_t>(idx[static_cast<size_t>(chosen)]);
}
} // namespace

Model::Model(const ModelMeta& meta, llaisysDeviceType_t device_type, int device_id)
    : meta_(meta), device_type_(device_type), device_id_(device_id), cache_len_(0) {
    
    // 初始化 KV Cache
    k_cache_.resize(meta_.nlayer);
    v_cache_.resize(meta_.nlayer);
    for (size_t i = 0; i < meta_.nlayer; ++i) {
        k_cache_[i] = Tensor::create({meta_.maxseq, meta_.nkvh, meta_.dh}, 
                                     meta_.dtype, device_type_, device_id_);
        v_cache_[i] = Tensor::create({meta_.maxseq, meta_.nkvh, meta_.dh}, 
                                     meta_.dtype, device_type_, device_id_);
    }
    
    // 初始化权重数组
    weights_.attn_norm_w.resize(meta_.nlayer);
    weights_.attn_q_w.resize(meta_.nlayer);
    weights_.attn_q_b.resize(meta_.nlayer);
    weights_.attn_k_w.resize(meta_.nlayer);
    weights_.attn_k_b.resize(meta_.nlayer);
    weights_.attn_v_w.resize(meta_.nlayer);
    weights_.attn_v_b.resize(meta_.nlayer);
    weights_.attn_o_w.resize(meta_.nlayer);
    weights_.mlp_norm_w.resize(meta_.nlayer);
    weights_.mlp_gate_w.resize(meta_.nlayer);
    weights_.mlp_up_w.resize(meta_.nlayer);
    weights_.mlp_down_w.resize(meta_.nlayer);
    
    // 创建 dummy bias tensors（全零，用于没有 bias 的层）
    dummy_bias_hs_ = Tensor::create({meta_.hs}, meta_.dtype, device_type_, device_id_);
    dummy_bias_di_ = Tensor::create({meta_.di}, meta_.dtype, device_type_, device_id_);
    dummy_bias_q_ = Tensor::create({meta_.nh * meta_.dh}, meta_.dtype, device_type_, device_id_);
    dummy_bias_kv_ = Tensor::create({meta_.nkvh * meta_.dh}, meta_.dtype, device_type_, device_id_);
    dummy_bias_voc_ = Tensor::create({meta_.voc}, meta_.dtype, device_type_, device_id_);

    // dummy bias 必须显式清零，否则会把未初始化内存当作 bias 加进去，导致输出完全错误
    auto zero_tensor = [](const tensor_t &t) {
        std::vector<std::byte> zeros(t->numel() * t->elementSize(), std::byte{0});
        t->load(zeros.data());
    };
    zero_tensor(dummy_bias_hs_);
    zero_tensor(dummy_bias_di_);
    zero_tensor(dummy_bias_q_);
    zero_tensor(dummy_bias_kv_);
    zero_tensor(dummy_bias_voc_);
}

Model::~Model() {
    // 智能指针会自动管理内存
}

void Model::reset_cache() {
    cache_len_ = 0;
}

void Model::ensure_tensor(tensor_t &tensor, const std::vector<size_t> &shape, llaisysDataType_t dtype) {
    const bool need_new = (!tensor)
                          || tensor->dtype() != dtype
                          || tensor->deviceType() != device_type_
                          || tensor->deviceId() != device_id_
                          || tensor->shape() != shape;
    if (need_new) {
        tensor = Tensor::create(shape, dtype, device_type_, device_id_);
    }
}

void Model::update_kv_cache(size_t layer_idx, tensor_t k_new, tensor_t v_new, size_t seqlen, size_t old_len) {
    // 将新的 K 和 V 追加到 cache
    // k_new: [seqlen, nkvh, dh]
    // v_new: [seqlen, nkvh, dh]

    // old_len 必须是"本次 forward 开始前"的 cache 长度。
    // 注意：cache_len_ 是全局序列长度，不应在每一层里自增。
    ASSERT(old_len == cache_len_, "update_kv_cache: old_len must equal cache_len_");
    size_t new_len = old_len + seqlen;
    CHECK_ARGUMENT(new_len <= meta_.maxseq, "update_kv_cache: cache overflow");
    
    // 复制新计算的 K 和 V 到 cache
    // 使用运行时 API 的内存拷贝，支持跨设备
    llaisys::core::context().setDevice(device_type_, device_id_);
    const LlaisysRuntimeAPI *api = llaisys::core::context().runtime().api();
    
    // 使用 tensor 的 numel 和 elementSize 计算正确的字节数
    size_t k_size = k_new->numel() * k_new->elementSize();
    size_t v_size = v_new->numel() * v_new->elementSize();
    
    // 确保 k_new 和 v_new 是连续的
    ASSERT(k_new->isContiguous() && v_new->isContiguous(), 
           "update_kv_cache: k_new and v_new must be contiguous");
    ASSERT(k_cache_[layer_idx]->isContiguous() && v_cache_[layer_idx]->isContiguous(),
           "update_kv_cache: cache tensors must be contiguous");
    
    // cache/new 都在同一设备上，使用 D2D
    const size_t cache_row_bytes = meta_.nkvh * meta_.dh * k_new->elementSize();
    const size_t dst_offset_bytes = old_len * cache_row_bytes;
    api->memcpy_sync(k_cache_[layer_idx]->data() + dst_offset_bytes, k_new->data(), k_size, LLAISYS_MEMCPY_D2D);
    api->memcpy_sync(v_cache_[layer_idx]->data() + dst_offset_bytes, v_new->data(), v_size, LLAISYS_MEMCPY_D2D);
}

void Model::forward_layer(size_t layer_idx, tensor_t& x, size_t seqlen, size_t total_len, tensor_t pos_ids_q) {
    // 设置设备上下文
    llaisys::core::context().setDevice(device_type_, device_id_);
    
    // 1. Pre-attention norm
    ensure_tensor(x_norm_, {seqlen, meta_.hs}, meta_.dtype);
    ops::rms_norm(x_norm_, x, weights_.attn_norm_w[layer_idx], meta_.epsilon);
    
    // 2. Attention
    // 2.1 计算 Q, K, V
    // x_norm: [seqlen, hs]
    // Q weight: [nh * dh, hs], output: [seqlen, nh * dh]
    // K weight: [nkvh * dh, hs], output: [seqlen, nkvh * dh]
    // V weight: [nkvh * dh, hs], output: [seqlen, nkvh * dh]
    
    ensure_tensor(q_flat_, {seqlen, meta_.nh * meta_.dh}, meta_.dtype);
    ensure_tensor(k_flat_, {seqlen, meta_.nkvh * meta_.dh}, meta_.dtype);
    ensure_tensor(v_flat_, {seqlen, meta_.nkvh * meta_.dh}, meta_.dtype);
    
    // 处理可能为空的 bias：如果不存在，使用 dummy bias
    tensor_t q_bias = (weights_.attn_q_b[layer_idx] && weights_.attn_q_b[layer_idx]->numel() > 0) ? 
                      weights_.attn_q_b[layer_idx] : dummy_bias_q_;
    tensor_t k_bias = (weights_.attn_k_b[layer_idx] && weights_.attn_k_b[layer_idx]->numel() > 0) ? 
                      weights_.attn_k_b[layer_idx] : dummy_bias_kv_;
    tensor_t v_bias = (weights_.attn_v_b[layer_idx] && weights_.attn_v_b[layer_idx]->numel() > 0) ? 
                      weights_.attn_v_b[layer_idx] : dummy_bias_kv_;
    
    ops::linear(q_flat_, x_norm_, weights_.attn_q_w[layer_idx], q_bias);
    ops::linear(k_flat_, x_norm_, weights_.attn_k_w[layer_idx], k_bias);
    ops::linear(v_flat_, x_norm_, weights_.attn_v_w[layer_idx], v_bias);
    
    // Reshape: [seqlen, nh * dh] -> [seqlen, nh, dh]
    q_ = q_flat_->view({seqlen, meta_.nh, meta_.dh});
    k_ = k_flat_->view({seqlen, meta_.nkvh, meta_.dh});
    v_ = v_flat_->view({seqlen, meta_.nkvh, meta_.dh});
    
    // 2.2 RoPE（只处理本轮新增 token）
    ensure_tensor(q_rope_, {seqlen, meta_.nh, meta_.dh}, meta_.dtype);
    ensure_tensor(k_rope_new_, {seqlen, meta_.nkvh, meta_.dh}, meta_.dtype);
    ops::rope(k_rope_new_, k_, pos_ids_q, meta_.theta);
    ops::rope(q_rope_, q_, pos_ids_q, meta_.theta);

    // 2.3 更新 KV Cache（K 使用 RoPE 后结果，V 保持原值）
    size_t old_len = total_len - seqlen;
    update_kv_cache(layer_idx, k_rope_new_, v_, seqlen, old_len);

    // 2.4 准备完整的 K 和 V（包含 cache）
    k_full_ = k_cache_[layer_idx]->slice(0, 0, total_len);
    v_full_ = v_cache_[layer_idx]->slice(0, 0, total_len);
    
    // 2.5 Self-attention
    ensure_tensor(attn_out_, {seqlen, meta_.nh, meta_.dh}, meta_.dtype);
    float scale = 1.0f / std::sqrt(static_cast<float>(meta_.dh));
    ops::self_attention(attn_out_, q_rope_, k_full_, v_full_, scale);
    
    // 2.6 Attention output projection
    // attn_out: [seqlen, nh, dh] -> [seqlen, nh * dh]
    tensor_t attn_out_flat = attn_out_->view({seqlen, meta_.nh * meta_.dh});
    ensure_tensor(attn_proj_out_, {seqlen, meta_.hs}, meta_.dtype);
    ops::linear(attn_proj_out_, attn_out_flat, weights_.attn_o_w[layer_idx], nullptr);
    
    // 2.7 残差连接
    ensure_tensor(x_attn_, {seqlen, meta_.hs}, meta_.dtype);
    ops::add(x_attn_, x, attn_proj_out_);
    x = x_attn_;
    
    // 3. Post-attention norm
    ensure_tensor(x_norm_, {seqlen, meta_.hs}, meta_.dtype);
    ops::rms_norm(x_norm_, x, weights_.mlp_norm_w[layer_idx], meta_.epsilon);
    
    // 4. MLP
    // x_norm: [seqlen, hs]
    ensure_tensor(gate_, {seqlen, meta_.di}, meta_.dtype);
    ensure_tensor(up_, {seqlen, meta_.di}, meta_.dtype);
    
    ops::linear(gate_, x_norm_, weights_.mlp_gate_w[layer_idx], nullptr);
    ops::linear(up_, x_norm_, weights_.mlp_up_w[layer_idx], nullptr);
    
    ensure_tensor(swiglu_out_, {seqlen, meta_.di}, meta_.dtype);
    ops::swiglu(swiglu_out_, gate_, up_);
    
    ensure_tensor(mlp_out_, {seqlen, meta_.hs}, meta_.dtype);
    ops::linear(mlp_out_, swiglu_out_, weights_.mlp_down_w[layer_idx], nullptr);
    
    // 5. 残差连接
    ensure_tensor(x_mlp_, {seqlen, meta_.hs}, meta_.dtype);
    ops::add(x_mlp_, x, mlp_out_);
    x = x_mlp_;
}

tensor_t Model::forward(tensor_t input_ids, size_t seqlen, size_t total_len) {
    // 设置设备上下文
    llaisys::core::context().setDevice(device_type_, device_id_);
    
    // 1. Embedding
    ensure_tensor(x_, {seqlen, meta_.hs}, meta_.dtype);
    ops::embedding(x_, input_ids, weights_.in_embed);
    
    // 2. 本轮所有层复用同一份 pos_ids（避免每层重复构造与拷贝）
    size_t start_pos = total_len - seqlen;
    ensure_tensor(pos_ids_q_, {seqlen}, LLAISYS_DTYPE_I64);
    if (seqlen == 1) {
        int64_t pos = static_cast<int64_t>(start_pos);
        pos_ids_q_->load(&pos);
    } else {
        std::vector<int64_t> pos_ids_q_host(seqlen);
        for (size_t i = 0; i < seqlen; ++i) {
            pos_ids_q_host[i] = static_cast<int64_t>(start_pos + i);
        }
        pos_ids_q_->load(pos_ids_q_host.data());
    }

    // 3. Transformer layers
    for (size_t i = 0; i < meta_.nlayer; ++i) {
        forward_layer(i, x_, seqlen, total_len, pos_ids_q_);
    }

    // 4. Output norm
    ensure_tensor(x_norm_, {seqlen, meta_.hs}, meta_.dtype);
    ops::rms_norm(x_norm_, x_, weights_.out_norm_w, meta_.epsilon);

    // 5. Output projection (logits)
    ensure_tensor(logits_, {seqlen, meta_.voc}, meta_.dtype);
    // out_embed 应该是 [voc, hs]，linear 计算 Y = X W^T，所以 Y = [seqlen, voc]
    ops::linear(logits_, x_norm_, weights_.out_embed, nullptr);
    
    return logits_;
}

int64_t Model::infer(
    int64_t* token_ids,
    size_t ntoken,
    int top_k,
    float top_p,
    float temperature) {
    // 设置设备上下文
    llaisys::core::context().setDevice(device_type_, device_id_);
    
    // 创建输入张量
    ensure_tensor(input_ids_buf_, {ntoken}, LLAISYS_DTYPE_I64);
    input_ids_buf_->load(token_ids);
    
    // 确定序列长度
    size_t seqlen = ntoken;
    size_t total_len = cache_len_ + seqlen;
    
    // 前向传播
    tensor_t logits = forward(input_ids_buf_, seqlen, total_len);

    // 本轮 forward 已把每层 K/V 写入 cache 的 [cache_len_, total_len) 区间
    cache_len_ = total_len;
    
    // 获取最后一个 token 的 logits
    tensor_t last_logits = logits->slice(0, seqlen - 1, seqlen);
    last_logits = last_logits->view({meta_.voc});
    
    const bool greedy = (top_k == 1) && (top_p >= 1.0f) && (std::abs(temperature - 1.0f) < 1e-6f);
    if (greedy) {
        // Fast path: keep current argmax operator pipeline.
        ensure_tensor(max_idx_, {1}, LLAISYS_DTYPE_I64);
        ensure_tensor(max_val_, {1}, meta_.dtype);
        ops::argmax(max_idx_, max_val_, last_logits);

        int64_t host_result = 0;
        llaisys::core::context().runtime().api()->memcpy_sync(
            &host_result, max_idx_->data(), sizeof(int64_t), LLAISYS_MEMCPY_D2H);
        return host_result;
    }

    // Sampling path: read last-step logits to host and apply top-k/top-p/temperature.
    const LlaisysRuntimeAPI *api = llaisys::core::context().runtime().api();
    std::vector<float> host_logits = logits_to_host_f32(last_logits, api);
    return sample_from_logits(host_logits, top_k, top_p, temperature);
}

} // namespace llaisys::models::qwen2
