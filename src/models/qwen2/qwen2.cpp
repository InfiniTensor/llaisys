#include "qwen2.hpp"
#include "../../utils.hpp"

namespace llaisys::models {
namespace {

std::pair<size_t, size_t> evenShardRange(size_t total, size_t rank, size_t world_size) {
    ASSERT(world_size > 0, "Tensor parallel world size must be positive.");
    ASSERT(total % world_size == 0, "Tensor parallel shard requires evenly divisible dimension.");
    const size_t shard = total / world_size;
    return {rank * shard, (rank + 1) * shard};
}

tensor_t sliceColumns(const tensor_t &tensor, size_t start, size_t end) {
    ASSERT(tensor->ndim() == 2, "sliceColumns expects a 2D tensor.");
    return tensor->slice(1, start, end);
}

tensor_t makeContiguous(const tensor_t &tensor) {
    if (tensor->isContiguous()) {
        return tensor;
    }
    auto out = Tensor::create(tensor->shape(), tensor->dtype(), tensor->deviceType(), tensor->deviceId());
    ops::rearrange(out, tensor);
    return out;
}

float readScalarAsFloat(const std::byte *data, llaisysDataType_t dtype, size_t index) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return reinterpret_cast<const float *>(data)[index];
    case LLAISYS_DTYPE_F16:
        return llaisys::utils::cast<float>(reinterpret_cast<const llaisys::fp16_t *>(data)[index]);
    case LLAISYS_DTYPE_BF16:
        return llaisys::utils::cast<float>(reinterpret_cast<const llaisys::bf16_t *>(data)[index]);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace

Qwen2Model::Qwen2Model(const Qwen2Config& config, llaisysDeviceType_t device_type, int device_id)
    : config_(config),
      device_type_(device_type),
      device_id_(device_id),
      tp_rank_(0),
      tp_world_size_(1),
      hidden_shard_size_(0),
      kv_hidden_size_(0),
      kv_shard_size_(0),
      intermediate_shard_size_(0),
      vocab_shard_size_(0),
      vocab_start_(0),
      vocab_end_(0),
      cache_len_(0),
      buffer_seq_len_(0) {
    if (device_type_ == LLAISYS_DEVICE_NVIDIA && core::context().distributedInitialized()) {
        tp_rank_ = static_cast<size_t>(core::context().distributedRank());
        tp_world_size_ = static_cast<size_t>(core::context().distributedWorldSize());
    }
    
    const size_t nl = config_.num_layers;
    const size_t hs = config_.hidden_size;
    kv_hidden_size_ = config_.num_kv_heads * config_.head_dim;
    ASSERT(hs % tp_world_size_ == 0, "Tensor parallel hidden size must be divisible by world size.");
    ASSERT(kv_hidden_size_ % tp_world_size_ == 0, "Tensor parallel KV size must be divisible by world size.");
    ASSERT(config_.intermediate_size % tp_world_size_ == 0, "Tensor parallel intermediate size must be divisible by world size.");
    ASSERT(config_.vocab_size % tp_world_size_ == 0, "Tensor parallel vocab size must be divisible by world size.");
    hidden_shard_size_ = hs / tp_world_size_;
    kv_shard_size_ = kv_hidden_size_ / tp_world_size_;
    intermediate_shard_size_ = config_.intermediate_size / tp_world_size_;
    vocab_shard_size_ = config_.vocab_size / tp_world_size_;
    auto vocab_range = evenShardRange(config_.vocab_size, tp_rank_, tp_world_size_);
    vocab_start_ = vocab_range.first;
    vocab_end_ = vocab_range.second;
    
    // Allocate embedding weights
    embed_tokens_ = Tensor::create({vocab_shard_size_, hs}, config_.dtype, device_type_, device_id_);
    lm_head_ = Tensor::create({vocab_shard_size_, hs}, config_.dtype, device_type_, device_id_);
    norm_weight_ = Tensor::create({hs}, config_.dtype, device_type_, device_id_);
    
    // Allocate per-layer weights
    input_layernorm_weight_.resize(nl);
    q_proj_weight_.resize(nl);
    q_proj_bias_.resize(nl);
    k_proj_weight_.resize(nl);
    k_proj_bias_.resize(nl);
    v_proj_weight_.resize(nl);
    v_proj_bias_.resize(nl);
    o_proj_weight_.resize(nl);
    post_attn_layernorm_weight_.resize(nl);
    gate_proj_weight_.resize(nl);
    up_proj_weight_.resize(nl);
    down_proj_weight_.resize(nl);
    
    for (size_t i = 0; i < nl; ++i) {
        input_layernorm_weight_[i] = Tensor::create({hs}, config_.dtype, device_type_, device_id_);
        q_proj_weight_[i] = Tensor::create({hidden_shard_size_, hs}, config_.dtype, device_type_, device_id_);
        q_proj_bias_[i] = Tensor::create({hidden_shard_size_}, config_.dtype, device_type_, device_id_);
        k_proj_weight_[i] = Tensor::create({kv_shard_size_, hs}, config_.dtype, device_type_, device_id_);
        k_proj_bias_[i] = Tensor::create({kv_shard_size_}, config_.dtype, device_type_, device_id_);
        v_proj_weight_[i] = Tensor::create({kv_shard_size_, hs}, config_.dtype, device_type_, device_id_);
        v_proj_bias_[i] = Tensor::create({kv_shard_size_}, config_.dtype, device_type_, device_id_);
        o_proj_weight_[i] = Tensor::create({hs, hidden_shard_size_}, config_.dtype, device_type_, device_id_);
        post_attn_layernorm_weight_[i] = Tensor::create({hs}, config_.dtype, device_type_, device_id_);
        gate_proj_weight_[i] = Tensor::create({intermediate_shard_size_, hs}, config_.dtype, device_type_, device_id_);
        up_proj_weight_[i] = Tensor::create({intermediate_shard_size_, hs}, config_.dtype, device_type_, device_id_);
        down_proj_weight_[i] = Tensor::create({hs, intermediate_shard_size_}, config_.dtype, device_type_, device_id_);
    }
    
    // Allocate KV cache
    allocateKVCache();
}

void Qwen2Model::allocateKVCache() {
    const size_t nl = config_.num_layers;
    const size_t maxseq = config_.max_seq_len;
    
    k_cache_.resize(nl);
    v_cache_.resize(nl);
    
    for (size_t i = 0; i < nl; ++i) {
        k_cache_[i] = Tensor::create({maxseq, kv_shard_size_}, config_.dtype, device_type_, device_id_);
        v_cache_[i] = Tensor::create({maxseq, kv_shard_size_}, config_.dtype, device_type_, device_id_);
    }
}

void Qwen2Model::allocateBuffers(size_t seq_len) {
    if (buffer_seq_len_ == seq_len && hidden_states_ != nullptr) {
        return;
    }

    const size_t hs = config_.hidden_size;
    
    hidden_states_ = Tensor::create({seq_len, hs}, config_.dtype, device_type_, device_id_);
    normed_ = Tensor::create({seq_len, hs}, config_.dtype, device_type_, device_id_);
    q_local_ = Tensor::create({seq_len, hidden_shard_size_}, config_.dtype, device_type_, device_id_);
    k_local_ = Tensor::create({seq_len, kv_shard_size_}, config_.dtype, device_type_, device_id_);
    v_local_ = Tensor::create({seq_len, kv_shard_size_}, config_.dtype, device_type_, device_id_);
    attn_proj_ = Tensor::create({seq_len, hs}, config_.dtype, device_type_, device_id_);
    gate_ = Tensor::create({seq_len, intermediate_shard_size_}, config_.dtype, device_type_, device_id_);
    up_ = Tensor::create({seq_len, intermediate_shard_size_}, config_.dtype, device_type_, device_id_);
    mlp_out_ = Tensor::create({seq_len, intermediate_shard_size_}, config_.dtype, device_type_, device_id_);
    down_ = Tensor::create({seq_len, hs}, config_.dtype, device_type_, device_id_);
    logits_local_ = Tensor::create({1, vocab_shard_size_}, config_.dtype, device_type_, device_id_);
    pos_ids_ = Tensor::create({seq_len}, LLAISYS_DTYPE_I64, device_type_, device_id_);
    input_ids_ = Tensor::create({seq_len}, LLAISYS_DTYPE_I64, device_type_, device_id_);
    max_idx_ = Tensor::create({1}, LLAISYS_DTYPE_I64, device_type_, device_id_);
    max_val_ = Tensor::create({1}, config_.dtype, device_type_, device_id_);
    buffer_seq_len_ = seq_len;
}

int64_t Qwen2Model::infer(const int64_t* token_ids, size_t num_tokens) {
    const size_t nh = config_.num_heads;
    const size_t dh = config_.head_dim;
    const size_t nl = config_.num_layers;
    const float eps = config_.rms_norm_eps;
    const float theta = config_.rope_theta;
    const float scale = 1.0f / std::sqrt(static_cast<float>(dh));
    
    // Allocate buffers for this sequence length
    allocateBuffers(num_tokens);
    
    // Create input token tensor
    input_ids_->load(token_ids);
    
    // Create position ids: [cache_len, cache_len+1, ..., cache_len+num_tokens-1]
    std::vector<int64_t> pos_ids_data(num_tokens);
    for (size_t i = 0; i < num_tokens; ++i) {
        pos_ids_data[i] = static_cast<int64_t>(cache_len_ + i);
    }
    pos_ids_->load(pos_ids_data.data());
    
    // Embedding lookup: hidden_states = embed_tokens[input_ids]
    ops::parallelEmbedding(hidden_states_, input_ids_, embed_tokens_, vocab_start_, vocab_end_);
    if (tp_world_size_ > 1) {
        core::context().allReduce(hidden_states_);
    }
    
    // Process each layer
    for (size_t layer = 0; layer < nl; ++layer) {
        // Input layernorm
        ops::rms_norm(normed_, hidden_states_, input_layernorm_weight_[layer], eps);
        
        ops::columnParallelLinear(q_local_, normed_, q_proj_weight_[layer], q_proj_bias_[layer]);
        ops::columnParallelLinear(k_local_, normed_, k_proj_weight_[layer], k_proj_bias_[layer]);
        ops::columnParallelLinear(v_local_, normed_, v_proj_weight_[layer], v_proj_bias_[layer]);

        auto q_full_flat = ops::gatherLastDim(q_local_);
        auto k_full_flat = ops::gatherLastDim(k_local_);
        auto q_full = q_full_flat->view({num_tokens, nh, dh});
        auto k_full = k_full_flat->view({num_tokens, config_.num_kv_heads, dh});

        ops::rope(q_full, q_full, pos_ids_, theta);
        ops::rope(k_full, k_full, pos_ids_, theta);
        
        // Update KV cache
        auto k_cache_slice = k_cache_[layer]->slice(0, cache_len_, cache_len_ + num_tokens);
        auto v_cache_slice = v_cache_[layer]->slice(0, cache_len_, cache_len_ + num_tokens);
        auto k_after_rope_flat = k_full->view({num_tokens, kv_hidden_size_});
        auto k_local_after_rope = makeContiguous(
            sliceColumns(k_after_rope_flat, tp_rank_ * kv_shard_size_, (tp_rank_ + 1) * kv_shard_size_));
        ops::rearrange(k_cache_slice, k_local_after_rope);
        ops::rearrange(v_cache_slice, v_local_);

        auto k_full_cache = ops::gatherLastDim(k_cache_[layer]->slice(0, 0, cache_len_ + num_tokens));
        auto v_full_cache = ops::gatherLastDim(v_cache_[layer]->slice(0, 0, cache_len_ + num_tokens));
        auto k_attn = k_full_cache->view({cache_len_ + num_tokens, config_.num_kv_heads, dh});
        auto v_attn = v_full_cache->view({cache_len_ + num_tokens, config_.num_kv_heads, dh});
        
        // Self attention
        auto attn_out = Tensor::create({num_tokens, nh, dh}, config_.dtype, device_type_, device_id_);
        ops::self_attention(attn_out, q_full, k_attn, v_attn, scale);
        
        // Output projection
        auto attn_out_flat = attn_out->view({num_tokens, nh * dh});
        auto attn_out_local = makeContiguous(
            sliceColumns(attn_out_flat, tp_rank_ * hidden_shard_size_, (tp_rank_ + 1) * hidden_shard_size_));
        ops::rowParallelLinear(attn_proj_, attn_out_local, o_proj_weight_[layer]);
        
        // Residual connection: hidden_states = hidden_states + attn_proj
        ops::add(hidden_states_, hidden_states_, attn_proj_);
        
        // Post attention layernorm
        ops::rms_norm(normed_, hidden_states_, post_attn_layernorm_weight_[layer], eps);
        
        // MLP: gate_proj, up_proj, swiglu, down_proj
        ops::columnParallelLinear(gate_, normed_, gate_proj_weight_[layer], nullptr);
        ops::columnParallelLinear(up_, normed_, up_proj_weight_[layer], nullptr);
        ops::swiglu(mlp_out_, gate_, up_);
        ops::rowParallelLinear(down_, mlp_out_, down_proj_weight_[layer]);
        
        // Residual connection: hidden_states = hidden_states + down
        ops::add(hidden_states_, hidden_states_, down_);
    }
    
    // Final layer norm
    ops::rms_norm(normed_, hidden_states_, norm_weight_, eps);
    
    // Get last token's hidden state for prediction
    auto last_hidden = normed_->slice(0, num_tokens - 1, num_tokens);  // [1, hs]
    
    auto& ctx = core::context();
    ctx.setDevice(device_type_, device_id_);
    auto& runtime = ctx.runtime();

    // LM head local shard
    ops::columnParallelLinear(logits_local_, last_hidden, lm_head_, nullptr);
    auto logits_local_1d = logits_local_->view({vocab_shard_size_});
    ops::argmax(max_idx_, max_val_, logits_local_1d);

    int64_t local_idx = 0;
    runtime.api()->memcpy_sync(&local_idx, max_idx_->data(), sizeof(int64_t), LLAISYS_MEMCPY_D2H);
    local_idx += static_cast<int64_t>(vocab_start_);

    int64_t next_token = local_idx;
    if (tp_world_size_ > 1) {
        auto local_global_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, device_type_, device_id_);
        local_global_idx->load(&local_idx);
        auto gathered_values = ctx.allGather(max_val_);
        auto gathered_indices = ctx.allGather(local_global_idx);

        std::vector<std::byte> gathered_value_bytes(gathered_values->numel() * gathered_values->elementSize());
        std::vector<int64_t> gathered_index_host(gathered_indices->numel());
        runtime.api()->memcpy_sync(gathered_value_bytes.data(), gathered_values->data(), gathered_value_bytes.size(), LLAISYS_MEMCPY_D2H);
        runtime.api()->memcpy_sync(gathered_index_host.data(), gathered_indices->data(),
                                   gathered_index_host.size() * sizeof(int64_t), LLAISYS_MEMCPY_D2H);

        float best_value = readScalarAsFloat(gathered_value_bytes.data(), config_.dtype, 0);
        next_token = gathered_index_host[0];
        for (size_t rank = 1; rank < tp_world_size_; ++rank) {
            const float value = readScalarAsFloat(gathered_value_bytes.data(), config_.dtype, rank);
            const int64_t index = gathered_index_host[rank];
            if (value > best_value || (value == best_value && index < next_token)) {
                best_value = value;
                next_token = index;
            }
        }
    }
    
    // Update cache length
    cache_len_ += num_tokens;
    
    return next_token;
}

} // namespace llaisys::models
