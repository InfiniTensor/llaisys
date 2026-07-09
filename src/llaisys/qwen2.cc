#include "llaisys/models/qwen2.h"
#include "llaisys_tensor.hpp"

#include "../tensor/tensor.hpp"
#include "../core/llaisys_core.hpp"
#include "../ops/add/op.hpp"
#include "../ops/rand_sample/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rearrange/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"
#include "../utils.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <unordered_map>
#include <vector>

// wrap c++ tensor to external handle
namespace {

// based on modelMetaInfo create tensor
llaisys::tensor_t make_tensor(
    const LlaisysQwen2Meta &meta,
    llaisysDeviceType_t device,
    int device_id,
    const std::vector<size_t> &shape) {
    return llaisys::Tensor::create(shape, meta.dtype, device, device_id);
}

// based on dtype create tensor
llaisys::tensor_t make_tensor_dtype(
    llaisysDataType_t dtype,
    llaisysDeviceType_t device,
    int device_id,
    const std::vector<size_t> &shape) {
    return llaisys::Tensor::create(shape, dtype, device, device_id);
}

// set tensor data to zero
void zero_tensor(const llaisys::tensor_t &t) {
    size_t size = t->numel() * t->elementSize();
    if (t->deviceType() == LLAISYS_DEVICE_CPU) {
        std::memset(t->data(), 0, size);
        return;
    }
    std::vector<uint8_t> zeros(size);
    t->load(zeros.data());
}

// wrap c++ tensor to external handle and record to handles list
llaisysTensor_t wrap_tensor(
    const llaisys::tensor_t &t,
    std::vector<LlaisysTensor *> &handles) {
    auto *h = new LlaisysTensor{t};
    handles.push_back(h);
    return h;
}

size_t ceil_div(size_t a, size_t b) {
    return (a + b - 1) / b;
}

size_t read_env_size_t(const char *name, size_t default_value) {
#ifdef _WIN32
    char *value = nullptr;
    size_t len = 0;
    if (_dupenv_s(&value, &len, name) != 0 || value == nullptr) {
        return default_value;
    }
    char *end = nullptr;
    unsigned long long parsed = std::strtoull(value, &end, 10);
    std::free(value);
    if (end == value || *end != '\0') {
        return default_value;
    }
    return static_cast<size_t>(parsed);
#else
    const char *value = std::getenv(name);
    if (!value) {
        return default_value;
    }
    char *end = nullptr;
    unsigned long long parsed = std::strtoull(value, &end, 10);
    if (end == value || *end != '\0') {
        return default_value;
    }
    return static_cast<size_t>(parsed);
#endif
}

uint64_t hash_mix(uint64_t h, int64_t v, uint64_t base) {
    return h * base + (static_cast<uint64_t>(v) + 0x9e3779b97f4a7c15ull);
}

struct KVPagePoolLayer {
    std::vector<llaisys::tensor_t> k_pages;
    std::vector<llaisys::tensor_t> v_pages;
    std::vector<size_t> refcnt;
    std::vector<uint64_t> last_access;
    std::vector<size_t> free_ids;
    std::vector<uint8_t> is_free;
};

class KVCachePool {
public:
    void init(const LlaisysQwen2Meta &meta, llaisysDeviceType_t device, int device_id, size_t page_len, size_t max_pages) {
        meta_ = meta;
        device_ = device;
        device_id_ = device_id;
        page_len_ = page_len;
        max_pages_ = max_pages;
        access_clock_ = 1;
        layers_.assign(meta.nlayer, KVPagePoolLayer{});
    }

    size_t page_len() const {
        return page_len_;
    }

    size_t acquire_page(size_t layer) {
        auto &pool = layers_[layer];
        while (!pool.free_ids.empty()) {
            size_t id = pool.free_ids.back();
            pool.free_ids.pop_back();
            if (pool.is_free[id] && pool.refcnt[id] == 0) {
                pool.is_free[id] = 0;
                pool.last_access[id] = access_clock_++;
                return id;
            }
        }

        if (pool.k_pages.size() < max_pages_) {
            size_t id = pool.k_pages.size();
            pool.k_pages.push_back(make_tensor(meta_, device_, device_id_, {page_len_, meta_.nkvh, meta_.dh}));
            pool.v_pages.push_back(make_tensor(meta_, device_, device_id_, {page_len_, meta_.nkvh, meta_.dh}));
            pool.refcnt.push_back(0);
            pool.last_access.push_back(access_clock_++);
            pool.is_free.push_back(0);
            return id;
        }

        size_t selected = std::numeric_limits<size_t>::max();
        uint64_t best_access = std::numeric_limits<uint64_t>::max();
        for (size_t i = 0; i < pool.k_pages.size(); ++i) {
            if (pool.refcnt[i] == 0 && pool.last_access[i] <= best_access) {
                best_access = pool.last_access[i];
                selected = i;
            }
        }
        CHECK_ARGUMENT(selected != std::numeric_limits<size_t>::max(), "KV cache pool has no free page.");
        pool.is_free[selected] = 0;
        pool.last_access[selected] = access_clock_++;
        return selected;
    }

    void incref(size_t layer, size_t page_id) {
        auto &pool = layers_[layer];
        if (pool.is_free[page_id]) {
            pool.is_free[page_id] = 0;
        }
        pool.refcnt[page_id] += 1;
        pool.last_access[page_id] = access_clock_++;
    }

    void decref(size_t layer, size_t page_id) {
        auto &pool = layers_[layer];
        if (pool.refcnt[page_id] > 0) {
            pool.refcnt[page_id] -= 1;
            if (pool.refcnt[page_id] == 0 && !pool.is_free[page_id]) {
                pool.is_free[page_id] = 1;
                pool.free_ids.push_back(page_id);
            }
        }
        pool.last_access[page_id] = access_clock_++;
    }

    llaisys::tensor_t k_page(size_t layer, size_t page_id) {
        auto &pool = layers_[layer];
        pool.last_access[page_id] = access_clock_++;
        return pool.k_pages[page_id];
    }

    llaisys::tensor_t v_page(size_t layer, size_t page_id) {
        auto &pool = layers_[layer];
        pool.last_access[page_id] = access_clock_++;
        return pool.v_pages[page_id];
    }

private:
    LlaisysQwen2Meta meta_;
    llaisysDeviceType_t device_;
    int device_id_;
    size_t page_len_;
    size_t max_pages_;
    uint64_t access_clock_ = 1;
    std::vector<KVPagePoolLayer> layers_;
};

struct KVHandle {
    std::vector<std::vector<size_t>> layer_pages;
    size_t token_count = 0;
    uint64_t last_access = 0;
    std::vector<int64_t> tokens;
    std::vector<uint64_t> hash_keys;
};

class PrefixCacheIndex {
public:
    void init(size_t max_handles) {
        max_handles_ = max_handles;
        access_clock_ = 1;
        handles_.clear();
        alive_.clear();
        key_to_handles_.clear();
        free_handle_ids_.clear();
    }

    size_t find_longest_prefix(const int64_t *tokens, size_t ntoken, const KVCachePool &pool, KVHandle &out_handle) {
        if (ntoken == 0) {
            return 0;
        }
        auto hashes = prefix_hashes(tokens, ntoken);
        for (size_t len = ntoken; len > 0; --len) {
            uint64_t key = make_key(hashes[len], len);
            auto it = key_to_handles_.find(key);
            if (it == key_to_handles_.end()) {
                continue;
            }
            for (size_t handle_id : it->second) {
                if (handle_id >= handles_.size() || !alive_[handle_id]) {
                    continue;
                }
                const auto &h = handles_[handle_id];
                if (h.tokens.size() < len) {
                    continue;
                }
                if (!std::equal(tokens, tokens + len, h.tokens.begin())) {
                    continue;
                }
                size_t pages_needed = ceil_div(len, pool.page_len());
                out_handle.layer_pages.assign(h.layer_pages.size(), {});
                for (size_t i = 0; i < h.layer_pages.size(); ++i) {
                    out_handle.layer_pages[i].assign(h.layer_pages[i].begin(), h.layer_pages[i].begin() + pages_needed);
                }
                out_handle.token_count = len;
                out_handle.tokens.assign(tokens, tokens + len);
                out_handle.last_access = access_clock_++;
                handles_[handle_id].last_access = out_handle.last_access;
                return len;
            }
        }
        return 0;
    }

    void insert_handle(const KVHandle &handle, const int64_t *tokens, size_t ntoken, KVCachePool &pool) {
        if (ntoken == 0) {
            return;
        }
        size_t handle_id = acquire_handle_id();
        if (handle_id >= handles_.size()) {
            handles_.resize(handle_id + 1);
            alive_.resize(handle_id + 1, 0);
        }
        if (alive_[handle_id]) {
            release_handle(handle_id, pool);
        }
        KVHandle stored;
        stored.token_count = ntoken;
        stored.tokens.assign(tokens, tokens + ntoken);
        stored.layer_pages = handle.layer_pages;
        size_t pages_needed = ceil_div(ntoken, pool.page_len());
        for (size_t i = 0; i < stored.layer_pages.size(); ++i) {
            if (stored.layer_pages[i].size() > pages_needed) {
                stored.layer_pages[i].resize(pages_needed);
            }
            for (size_t page_id : stored.layer_pages[i]) {
                pool.incref(i, page_id);
            }
        }
        auto hashes = prefix_hashes(tokens, ntoken);
        stored.hash_keys.reserve(ntoken);
        for (size_t len = 1; len <= ntoken; ++len) {
            uint64_t key = make_key(hashes[len], len);
            key_to_handles_[key].push_back(handle_id);
            stored.hash_keys.push_back(key);
        }
        stored.last_access = access_clock_++;
        handles_[handle_id] = std::move(stored);
        alive_[handle_id] = 1;
        enforce_capacity(pool);
    }

    void release_handle(size_t handle_id, KVCachePool &pool) {
        if (handle_id >= handles_.size() || !alive_[handle_id]) {
            return;
        }
        auto &h = handles_[handle_id];
        for (size_t i = 0; i < h.layer_pages.size(); ++i) {
            for (size_t page_id : h.layer_pages[i]) {
                pool.decref(i, page_id);
            }
        }
        for (uint64_t key : h.hash_keys) {
            auto it = key_to_handles_.find(key);
            if (it == key_to_handles_.end()) {
                continue;
            }
            auto &vec = it->second;
            vec.erase(std::remove(vec.begin(), vec.end(), handle_id), vec.end());
            if (vec.empty()) {
                key_to_handles_.erase(it);
            }
        }
        h.layer_pages.clear();
        h.tokens.clear();
        h.hash_keys.clear();
        alive_[handle_id] = 0;
        free_handle_ids_.push_back(handle_id);
    }

private:
    uint64_t make_key(uint64_t hash, size_t len) const {
        return hash ^ (salt_ * static_cast<uint64_t>(len));
    }

    std::vector<uint64_t> prefix_hashes(const int64_t *tokens, size_t ntoken) const {
        std::vector<uint64_t> hashes(ntoken + 1);
        uint64_t h = 0;
        for (size_t i = 0; i < ntoken; ++i) {
            h = hash_mix(h, tokens[i], base_);
            hashes[i + 1] = h;
        }
        return hashes;
    }

    size_t acquire_handle_id() {
        if (!free_handle_ids_.empty()) {
            size_t id = free_handle_ids_.back();
            free_handle_ids_.pop_back();
            return id;
        }
        return handles_.size();
    }

    void enforce_capacity(KVCachePool &pool) {
        if (max_handles_ == 0) {
            return;
        }
        size_t alive_count = 0;
        for (uint8_t v : alive_) {
            alive_count += v;
        }
        while (alive_count > max_handles_) {
            size_t oldest = std::numeric_limits<size_t>::max();
            uint64_t oldest_access = std::numeric_limits<uint64_t>::max();
            for (size_t i = 0; i < handles_.size(); ++i) {
                if (!alive_[i]) {
                    continue;
                }
                if (handles_[i].last_access <= oldest_access) {
                    oldest_access = handles_[i].last_access;
                    oldest = i;
                }
            }
            if (oldest == std::numeric_limits<size_t>::max()) {
                break;
            }
            release_handle(oldest, pool);
            alive_count -= 1;
        }
    }

    uint64_t base_ = 1469598103934665603ull;
    uint64_t salt_ = 1099511628211ull;
    size_t max_handles_ = 64;
    uint64_t access_clock_ = 1;
    std::unordered_map<uint64_t, std::vector<size_t>> key_to_handles_;
    std::vector<KVHandle> handles_;
    std::vector<uint8_t> alive_;
    std::vector<size_t> free_handle_ids_;
};

}

// Qwen2 Model Instance Structure
struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta;              // model meta info
    llaisysDeviceType_t device;         // device type 
    int device_id;                      // device id

    LlaisysQwen2Weights weights;        // all weight tensors handle
    std::vector<LlaisysTensor *> handles; // handles list for unified release

    // attn out bias
    std::vector<llaisys::tensor_t> attn_o_bias;   
    // MLP gate bias
    std::vector<llaisys::tensor_t> mlp_gate_bias; 
    // MLP up bias
    std::vector<llaisys::tensor_t> mlp_up_bias;   
    // MLP down bias
    std::vector<llaisys::tensor_t> mlp_down_bias; 
    // out bias
    llaisys::tensor_t out_bias;                  

    // KV cache
    KVCachePool kv_pool;
    PrefixCacheIndex prefix_index;
    KVHandle active_handle;
    bool active_valid;
    size_t cache_len;
};

static void release_active_handle(LlaisysQwen2Model *model) {
    if (!model || !model->active_valid) {
        return;
    }
    for (size_t i = 0; i < model->active_handle.layer_pages.size(); ++i) {
        for (size_t page_id : model->active_handle.layer_pages[i]) {
            model->kv_pool.decref(i, page_id);
        }
    }
    model->active_handle.layer_pages.assign(model->meta.nlayer, {});
    model->active_handle.tokens.clear();
    model->active_handle.token_count = 0;
    model->active_valid = false;
}

// init model weights, allocate memory and zero bias
static void init_weights(LlaisysQwen2Model *model) {
    const auto &m = model->meta;
    // input embedding
    model->weights.in_embed = wrap_tensor(
        make_tensor(m, model->device, model->device_id, {m.voc, m.hs}),
        model->handles);
    // output embedding
    model->weights.out_embed = wrap_tensor(
        make_tensor(m, model->device, model->device_id, {m.voc, m.hs}),
        model->handles);
    // output layer norm
    model->weights.out_norm_w = wrap_tensor(
        make_tensor(m, model->device, model->device_id, {m.hs}),
        model->handles);

    // allocate ptr array for each layer
    model->weights.attn_norm_w = new llaisysTensor_t[m.nlayer];
    model->weights.attn_q_w = new llaisysTensor_t[m.nlayer];
    model->weights.attn_q_b = new llaisysTensor_t[m.nlayer];
    model->weights.attn_k_w = new llaisysTensor_t[m.nlayer];
    model->weights.attn_k_b = new llaisysTensor_t[m.nlayer];
    model->weights.attn_v_w = new llaisysTensor_t[m.nlayer];
    model->weights.attn_v_b = new llaisysTensor_t[m.nlayer];
    model->weights.attn_o_w = new llaisysTensor_t[m.nlayer];
    model->weights.mlp_norm_w = new llaisysTensor_t[m.nlayer];
    model->weights.mlp_gate_w = new llaisysTensor_t[m.nlayer];
    model->weights.mlp_up_w = new llaisysTensor_t[m.nlayer];
    model->weights.mlp_down_w = new llaisysTensor_t[m.nlayer];

    // bias
    model->attn_o_bias.resize(m.nlayer);
    model->mlp_gate_bias.resize(m.nlayer);
    model->mlp_up_bias.resize(m.nlayer);
    model->mlp_down_bias.resize(m.nlayer);

    // for each layer, create its weight and bias tensor
    // first weight tensor
    for (size_t i = 0; i < m.nlayer; ++i) {
        // attn layer norm
        model->weights.attn_norm_w[i] = wrap_tensor(
            make_tensor(m, model->device, model->device_id, {m.hs}),
            model->handles);
        // Q/K/V/O proj weight and bias
        model->weights.attn_q_w[i] = wrap_tensor(
            make_tensor(m, model->device, model->device_id, {m.nh * m.dh, m.hs}),
            model->handles);
        model->weights.attn_q_b[i] = wrap_tensor(
            make_tensor(m, model->device, model->device_id, {m.nh * m.dh}),
            model->handles);
        model->weights.attn_k_w[i] = wrap_tensor(
            make_tensor(m, model->device, model->device_id, {m.nkvh * m.dh, m.hs}),
            model->handles);
        model->weights.attn_k_b[i] = wrap_tensor(
            make_tensor(m, model->device, model->device_id, {m.nkvh * m.dh}),
            model->handles);
        model->weights.attn_v_w[i] = wrap_tensor(
            make_tensor(m, model->device, model->device_id, {m.nkvh * m.dh, m.hs}),
            model->handles);
        model->weights.attn_v_b[i] = wrap_tensor(
            make_tensor(m, model->device, model->device_id, {m.nkvh * m.dh}),
            model->handles);
        model->weights.attn_o_w[i] = wrap_tensor(
            make_tensor(m, model->device, model->device_id, {m.hs, m.nh * m.dh}),
            model->handles);

        // MLP layer norm and gate/up/down weight
        model->weights.mlp_norm_w[i] = wrap_tensor(
            make_tensor(m, model->device, model->device_id, {m.hs}),
            model->handles);
        model->weights.mlp_gate_w[i] = wrap_tensor(
            make_tensor(m, model->device, model->device_id, {m.di, m.hs}),
            model->handles);
        model->weights.mlp_up_w[i] = wrap_tensor(
            make_tensor(m, model->device, model->device_id, {m.di, m.hs}),
            model->handles);
        model->weights.mlp_down_w[i] = wrap_tensor(
            make_tensor(m, model->device, model->device_id, {m.hs, m.di}),
            model->handles);

        // bias tensor
        model->attn_o_bias[i] = make_tensor(m, model->device, model->device_id, {m.hs});
        model->mlp_gate_bias[i] = make_tensor(m, model->device, model->device_id, {m.di});
        model->mlp_up_bias[i] = make_tensor(m, model->device, model->device_id, {m.di});
        model->mlp_down_bias[i] = make_tensor(m, model->device, model->device_id, {m.hs});

        // initialize bias tensor to 0
        zero_tensor(model->weights.attn_q_b[i]->tensor);
        zero_tensor(model->weights.attn_k_b[i]->tensor);
        zero_tensor(model->weights.attn_v_b[i]->tensor);
        zero_tensor(model->attn_o_bias[i]);
        zero_tensor(model->mlp_gate_bias[i]);
        zero_tensor(model->mlp_up_bias[i]);
        zero_tensor(model->mlp_down_bias[i]);
    }

    // output layerbias
    model->out_bias = make_tensor(m, model->device, model->device_id, {m.voc});
    zero_tensor(model->out_bias);
}

// initialize kv cache tensor
static void init_cache(LlaisysQwen2Model *model) {
    const auto &m = model->meta;
    size_t page_len = read_env_size_t("LLAISYS_KV_PAGE_LEN", 128);
    size_t max_pages = read_env_size_t("LLAISYS_KV_MAX_PAGES", ceil_div(m.maxseq, page_len));
    size_t max_handles = read_env_size_t("LLAISYS_KV_MAX_HANDLES", 64);
    model->kv_pool.init(m, model->device, model->device_id, page_len, max_pages);
    model->prefix_index.init(max_handles);
    model->active_handle.layer_pages.assign(m.nlayer, {});
    model->active_handle.tokens.clear();
    model->active_handle.token_count = 0;
    model->active_valid = false;
    model->cache_len = 0;
}

// inference implementation
static int64_t infer_impl(LlaisysQwen2Model *model, const int64_t *token_ids, size_t ntoken, float temperature,
                          size_t topK, float topP, int64_t seed) {
    CHECK_ARGUMENT(model != nullptr, "model is null");
    CHECK_ARGUMENT(token_ids != nullptr || ntoken == 0, "token_ids is null");
    CHECK_ARGUMENT(model->device == LLAISYS_DEVICE_CPU || model->device == LLAISYS_DEVICE_NVIDIA,
                   "Unsupported device type.");

    if (ntoken == 0) {
        return model->meta.end_token;
    }

    size_t reuse_len = 0;
    if (ntoken > 1) {
        release_active_handle(model);
        reuse_len = model->prefix_index.find_longest_prefix(token_ids, ntoken, model->kv_pool, model->active_handle);
        if (reuse_len > 0) {
            model->active_valid = true;
            for (size_t i = 0; i < model->active_handle.layer_pages.size(); ++i) {
                for (size_t page_id : model->active_handle.layer_pages[i]) {
                    model->kv_pool.incref(i, page_id);
                }
            }
        }
        if (reuse_len >= ntoken) {
            reuse_len = ntoken - 1;
        }
        if (model->active_valid) {
            size_t max_pages = ceil_div(reuse_len, model->kv_pool.page_len());
            for (size_t i = 0; i < model->active_handle.layer_pages.size(); ++i) {
                while (model->active_handle.layer_pages[i].size() > max_pages) {
                    size_t page_id = model->active_handle.layer_pages[i].back();
                    model->active_handle.layer_pages[i].pop_back();
                    model->kv_pool.decref(i, page_id);
                }
            }
            model->active_handle.token_count = reuse_len;
            model->active_handle.tokens.assign(token_ids, token_ids + reuse_len);
        }
        model->cache_len = reuse_len;
    }

    size_t seqlen = ntoken - reuse_len;
    size_t pos_offset = model->cache_len;

    // position ID [pos_offset, pos_offset + seqlen)
    std::vector<int64_t> pos_ids(seqlen);
    for (size_t i = 0; i < seqlen; ++i) {
        pos_ids[i] = static_cast<int64_t>(pos_offset + i);
    }

    // input token and position ID tensors
    auto input_ids = make_tensor_dtype(LLAISYS_DTYPE_I64, model->device, model->device_id, {seqlen});
    input_ids->load(token_ids + reuse_len);

    auto pos_tensor = make_tensor_dtype(LLAISYS_DTYPE_I64, model->device, model->device_id, {seqlen});
    pos_tensor->load(pos_ids.data());

    // embedding lookup
    auto x = make_tensor(model->meta, model->device, model->device_id, {seqlen, model->meta.hs});
    llaisys::ops::embedding(x, input_ids, model->weights.in_embed->tensor);

    // attn scale factor
    float scale = 1.0f / std::sqrt(static_cast<float>(model->meta.dh));

    size_t total_len = model->cache_len + seqlen;
    size_t page_len = model->kv_pool.page_len();
    size_t pages_needed = ceil_div(total_len, page_len);
    if (model->active_handle.layer_pages.size() != model->meta.nlayer) {
        model->active_handle.layer_pages.assign(model->meta.nlayer, {});
    }
    for (size_t i = 0; i < model->meta.nlayer; ++i) {
        while (model->active_handle.layer_pages[i].size() < pages_needed) {
            size_t page_id = model->kv_pool.acquire_page(i);
            model->kv_pool.incref(i, page_id);
            model->active_handle.layer_pages[i].push_back(page_id);
            model->active_valid = true;
        }
    }

    // layer forward
    for (size_t i = 0; i < model->meta.nlayer; ++i) {
        // attn input norm
        auto x_norm = make_tensor(model->meta, model->device, model->device_id, {seqlen, model->meta.hs});
        llaisys::ops::rms_norm(x_norm, x, model->weights.attn_norm_w[i]->tensor, model->meta.epsilon);

        // Q/K/V linear proj
        auto q = make_tensor(model->meta, model->device, model->device_id, {seqlen, model->meta.nh * model->meta.dh});
        auto k = make_tensor(model->meta, model->device, model->device_id, {seqlen, model->meta.nkvh * model->meta.dh});
        auto v = make_tensor(model->meta, model->device, model->device_id, {seqlen, model->meta.nkvh * model->meta.dh});

        llaisys::ops::linear(q, x_norm, model->weights.attn_q_w[i]->tensor, model->weights.attn_q_b[i]->tensor);
        llaisys::ops::linear(k, x_norm, model->weights.attn_k_w[i]->tensor, model->weights.attn_k_b[i]->tensor);
        llaisys::ops::linear(v, x_norm, model->weights.attn_v_w[i]->tensor, model->weights.attn_v_b[i]->tensor);

        // transform to multi-head dim
        auto q_view = q->view({seqlen, model->meta.nh, model->meta.dh});
        auto k_view = k->view({seqlen, model->meta.nkvh, model->meta.dh});
        auto v_view = v->view({seqlen, model->meta.nkvh, model->meta.dh});

        // RoPE
        auto q_rope = make_tensor(model->meta, model->device, model->device_id, {seqlen, model->meta.nh, model->meta.dh});
        auto k_rope = make_tensor(model->meta, model->device, model->device_id, {seqlen, model->meta.nkvh, model->meta.dh});

        llaisys::ops::rope(q_rope, q_view, pos_tensor, model->meta.theta);
        llaisys::ops::rope(k_rope, k_view, pos_tensor, model->meta.theta);

        size_t write_offset = model->cache_len;
        size_t remaining = seqlen;
        size_t src_offset = 0;
        while (remaining > 0) {
            size_t page_index = write_offset / page_len;
            size_t page_offset = write_offset % page_len;
            size_t chunk = std::min(remaining, page_len - page_offset);
            size_t page_id = model->active_handle.layer_pages[i][page_index];
            auto k_page = model->kv_pool.k_page(i, page_id);
            auto v_page = model->kv_pool.v_page(i, page_id);
            auto k_page_slice = k_page->slice(0, page_offset, page_offset + chunk);
            auto v_page_slice = v_page->slice(0, page_offset, page_offset + chunk);
            auto k_chunk = k_rope->slice(0, src_offset, src_offset + chunk);
            auto v_chunk = v_view->slice(0, src_offset, src_offset + chunk);
            llaisys::ops::rearrange(k_page_slice, k_chunk);
            llaisys::ops::rearrange(v_page_slice, v_chunk);
            write_offset += chunk;
            src_offset += chunk;
            remaining -= chunk;
        }

        auto k_total = make_tensor(model->meta, model->device, model->device_id, {total_len, model->meta.nkvh, model->meta.dh});
        auto v_total = make_tensor(model->meta, model->device, model->device_id, {total_len, model->meta.nkvh, model->meta.dh});
        size_t read_offset = 0;
        for (size_t page_index = 0; page_index < pages_needed; ++page_index) {
            size_t chunk = std::min(page_len, total_len - read_offset);
            size_t page_id = model->active_handle.layer_pages[i][page_index];
            auto k_page = model->kv_pool.k_page(i, page_id);
            auto v_page = model->kv_pool.v_page(i, page_id);
            auto k_page_slice = k_page->slice(0, 0, chunk);
            auto v_page_slice = v_page->slice(0, 0, chunk);
            auto k_total_slice = k_total->slice(0, read_offset, read_offset + chunk);
            auto v_total_slice = v_total->slice(0, read_offset, read_offset + chunk);
            llaisys::ops::rearrange(k_total_slice, k_page_slice);
            llaisys::ops::rearrange(v_total_slice, v_page_slice);
            read_offset += chunk;
        }

        // self attn
        auto attn = make_tensor(model->meta, model->device, model->device_id, {seqlen, model->meta.nh, model->meta.dh});
        llaisys::ops::self_attention(attn, q_rope, k_total, v_total, scale);

        // attn out proj
        auto attn_flat = attn->view({seqlen, model->meta.nh * model->meta.dh});
        auto attn_proj = make_tensor(model->meta, model->device, model->device_id, {seqlen, model->meta.hs});
        llaisys::ops::linear(attn_proj, attn_flat, model->weights.attn_o_w[i]->tensor, model->attn_o_bias[i]);

        // first residual conn
        auto res1 = make_tensor(model->meta, model->device, model->device_id, {seqlen, model->meta.hs});
        llaisys::ops::add(res1, x, attn_proj);

        // MLP input norm
        auto x_norm2 = make_tensor(model->meta, model->device, model->device_id, {seqlen, model->meta.hs});
        llaisys::ops::rms_norm(x_norm2, res1, model->weights.mlp_norm_w[i]->tensor, model->meta.epsilon);

        // MLP gate and up proj
        auto gate = make_tensor(model->meta, model->device, model->device_id, {seqlen, model->meta.di});
        auto up = make_tensor(model->meta, model->device, model->device_id, {seqlen, model->meta.di});

        llaisys::ops::linear(gate, x_norm2, model->weights.mlp_gate_w[i]->tensor, model->mlp_gate_bias[i]);
        llaisys::ops::linear(up, x_norm2, model->weights.mlp_up_w[i]->tensor, model->mlp_up_bias[i]);

        // SwiGLU activate
        auto swiglu_out = make_tensor(model->meta, model->device, model->device_id, {seqlen, model->meta.di});
        llaisys::ops::swiglu(swiglu_out, gate, up);

        // MLP down proj
        auto down = make_tensor(model->meta, model->device, model->device_id, {seqlen, model->meta.hs});
        llaisys::ops::linear(down, swiglu_out, model->weights.mlp_down_w[i]->tensor, model->mlp_down_bias[i]);

        // second residual conn
        auto res2 = make_tensor(model->meta, model->device, model->device_id, {seqlen, model->meta.hs});
        llaisys::ops::add(res2, res1, down);

        x = res2;
    }

    // update cache len
    model->cache_len += seqlen;
    model->active_handle.token_count = model->cache_len;
    if (model->active_valid) {
        if (model->active_handle.tokens.size() < model->cache_len) {
            model->active_handle.tokens.insert(
                model->active_handle.tokens.end(),
                token_ids + reuse_len,
                token_ids + reuse_len + seqlen);
        }
    }
    if (ntoken > 1 && model->active_valid) {
        model->prefix_index.insert_handle(model->active_handle, token_ids, ntoken, model->kv_pool);
    }

    // final norm
    auto x_norm = make_tensor(model->meta, model->device, model->device_id, {seqlen, model->meta.hs});
    llaisys::ops::rms_norm(x_norm, x, model->weights.out_norm_w->tensor, model->meta.epsilon);

    // output proj to vocab
    auto logits = make_tensor(model->meta, model->device, model->device_id, {seqlen, model->meta.voc});
    llaisys::ops::linear(logits, x_norm, model->weights.out_embed->tensor, model->out_bias);

    // get last token logits and sample
    auto last = logits->slice(0, seqlen - 1, seqlen)->view({model->meta.voc});
    auto sample_idx = make_tensor_dtype(LLAISYS_DTYPE_I64, model->device, model->device_id, {1});
    auto sample_val = make_tensor(model->meta, model->device, model->device_id, {1});
    llaisys::ops::rand_sample(sample_idx, sample_val, last, temperature, topK, topP, seed);

    if (model->device == LLAISYS_DEVICE_CPU) {
        return reinterpret_cast<int64_t *>(sample_idx->data())[0];
    }
    int64_t host_value = 0;
    llaisys::core::context().setDevice(model->device, model->device_id);
    llaisys::core::context().runtime().api()->memcpy_sync(
        &host_value,
        sample_idx->data(),
        sizeof(int64_t),
        LLAISYS_MEMCPY_D2H);
    return host_value;
}

// C API wrapper
__C {

// ModelCreate: Check args, initialize weights and cache
struct LlaisysQwen2Model *llaisysQwen2ModelCreate(
    const LlaisysQwen2Meta *meta,
    llaisysDeviceType_t device,
    int *device_ids,
    int ndevice) {

    auto *model = new LlaisysQwen2Model();
    model->meta = *meta;
    model->device = device;
    model->device_id = device_ids ? device_ids[0] : 0;
    init_weights(model);
    init_cache(model);
    return model;
}

// ModelDestroy: Free weights and cache, all handles
void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
    if (!model) {
        return;
    }
    delete[] model->weights.attn_norm_w;
    delete[] model->weights.attn_q_w;
    delete[] model->weights.attn_q_b;
    delete[] model->weights.attn_k_w;
    delete[] model->weights.attn_k_b;
    delete[] model->weights.attn_v_w;
    delete[] model->weights.attn_v_b;
    delete[] model->weights.attn_o_w;
    delete[] model->weights.mlp_norm_w;
    delete[] model->weights.mlp_gate_w;
    delete[] model->weights.mlp_up_w;
    delete[] model->weights.mlp_down_w;

    for (auto *h : model->handles) {
        delete h;
    }
    delete model;
}

// ModelWeights: Get weights pointer for loading pretrained params
struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
    return &model->weights;
}

// ModelInfer: Single interface for single token prediction
int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken, float temperature,
                               size_t topK, float topP, int64_t seed) {
    return infer_impl(model, token_ids, ntoken, temperature, topK, topP, seed);
}

}
