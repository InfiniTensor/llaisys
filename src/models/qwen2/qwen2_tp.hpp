#pragma once

#include "llaisys/models/qwen2_tp.h"
#include "qwen2.hpp"
#include "../../llaisys/llaisys_tensor.hpp"
#include "../../device/nvidia/nccl_communicator.hpp"

#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <future>

namespace llaisys {
namespace models {

// Simple barrier for C++17
class SimpleBarrier {
public:
    explicit SimpleBarrier(size_t count) : count_(count), generation_(0), waiting_(0) {}
    
    void arrive_and_wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        size_t gen = generation_;
        if (++waiting_ == count_) {
            generation_++;
            waiting_ = 0;
            cv_.notify_all();
        } else {
            cv_.wait(lock, [this, gen] { return gen != generation_; });
        }
    }
    
private:
    std::mutex mutex_;
    std::condition_variable cv_;
    size_t count_;
    size_t generation_;
    size_t waiting_;
};

// Model shard for a single GPU
struct Qwen2ModelShard {
    int rank;
    int device_id;
    
    // NCCL communicator for this rank
    device::nvidia::NCCLCommunicator* communicator;

    // Weight shards
    tensor_t in_embed;
    tensor_t out_embed;
    tensor_t out_norm_w;
    std::vector<tensor_t> attn_norm_w;
    std::vector<tensor_t> attn_q_w;
    std::vector<tensor_t> attn_q_b;
    std::vector<tensor_t> attn_k_w;
    std::vector<tensor_t> attn_k_b;
    std::vector<tensor_t> attn_v_w;
    std::vector<tensor_t> attn_v_b;
    std::vector<tensor_t> attn_o_w;
    std::vector<tensor_t> mlp_norm_w;
    std::vector<tensor_t> mlp_gate_w;
    std::vector<tensor_t> mlp_up_w;
    std::vector<tensor_t> mlp_down_w;

    // KV Cache
    std::vector<tensor_t> k_cache;
    std::vector<tensor_t> v_cache;

    // Intermediate buffers
    tensor_t hidden;
    tensor_t hidden_norm;
    tensor_t q;
    tensor_t k;
    tensor_t v;
    tensor_t q_rope;
    tensor_t k_rope;
    tensor_t attn_out;
    tensor_t attn_proj;
    tensor_t gate;
    tensor_t up;
    tensor_t mlp_out;
    tensor_t logits;
    tensor_t max_idx;
    tensor_t max_val;
    tensor_t pos_ids;
    tensor_t allreduce_buffer;

    Qwen2ModelShard(int rank_, int device_id_);

    void allocateWeights(const LlaisysQwen2Meta& meta, int world_size);
    void allocateCache(const LlaisysQwen2Meta& meta, int world_size);
    void allocateBuffers(const LlaisysQwen2Meta& meta, size_t max_seqlen, int world_size);

    size_t getShardedAttentionHeads(size_t nh, int world_size) const {
        return nh / world_size;
    }
    size_t getShardedKVHeads(size_t nkvh, int world_size) const {
        return nkvh / world_size;
    }
    size_t getShardedIntermediateSize(size_t di, int world_size) const {
        return di / world_size;
    }
};

// Inference task for worker threads
struct InferTask {
    int64_t* token_ids;
    size_t ntoken;
    size_t cache_len;
    std::promise<int64_t>* result_promise;  // Only rank 0 sets this
};

// Tensor Parallel Qwen2 Model
struct Qwen2ModelTP {
    LlaisysQwen2Meta meta;
    int world_size;
    std::vector<int> device_ids;

    // NCCL communicators (created in main thread, used by worker threads)
    std::vector<std::shared_ptr<device::nvidia::NCCLCommunicator>> communicators;

    // Model shards
    std::vector<std::unique_ptr<Qwen2ModelShard>> shards;

    // Current cache length
    std::atomic<size_t> cache_len;

    // Persistent worker threads
    std::vector<std::thread> worker_threads;
    std::atomic<bool> workers_running;
    
    // Task queue for inference
    std::vector<std::queue<InferTask>> task_queues;
    std::vector<std::mutex> queue_mutexes;
    std::vector<std::condition_variable> queue_cvs;
    
    // Completion synchronization
    SimpleBarrier* infer_barrier;
    std::atomic<int64_t> infer_result;

    Qwen2ModelTP(const LlaisysQwen2Meta* meta_, const std::vector<int>& device_ids);
    ~Qwen2ModelTP();

    void initialize();
    void startWorkerThreads();
    void stopWorkerThreads();
    void workerLoop(int rank);

    LlaisysQwen2Weights* getWeights(int rank);

    int64_t infer(int64_t* token_ids, size_t ntoken);
    void resetCache();
    int getWorldSize() const { return world_size; }

private:
    void forwardLayerWithBarrier(int rank, size_t layer, size_t seqlen, size_t start_pos, SimpleBarrier& barrier);
    void allReduce(int rank, void* buffer, size_t count, int dtype);
};

} // namespace models
} // namespace llaisys
