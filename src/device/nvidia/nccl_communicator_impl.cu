#include "nccl_communicator.hpp"
#include "nccl_communicator_impl.cuh"
#include <cstring>
#include <vector>
#include <thread>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace llaisys {
namespace device {
namespace nvidia {

// Impl structure definition
struct NCCLCommunicator::Impl {
    ncclComm_t comm;
    LlaisysCudaStream_t stream;
};

// Simple barrier implementation for C++17
class SimpleBarrier {
public:
    explicit SimpleBarrier(int count) : threshold_(count), count_(count), generation_(0) {}
    
    void arrive_and_wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        int gen = generation_;
        if (--count_ == 0) {
            generation_++;
            count_ = threshold_;
            cv_.notify_all();
        } else {
            cv_.wait(lock, [this, gen] { return gen != generation_; });
        }
    }
    
private:
    std::mutex mutex_;
    std::condition_variable cv_;
    int threshold_;
    int count_;
    int generation_;
};

// Static method to create all communicators
std::vector<std::shared_ptr<NCCLCommunicator>> NCCLCommunicator::createAll(
    const std::vector<int>& device_ids) {
    
    int world_size = static_cast<int>(device_ids.size());
    if (world_size == 0) {
        throw std::invalid_argument("Device IDs cannot be empty");
    }

    // Initialize CUDA driver
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    
    // Initialize CUDA context on first device
    CUDA_CHECK(cudaSetDevice(device_ids[0]));
    // Force driver initialization
    cudaFree(0);
    CUDA_CHECK(cudaDeviceSynchronize());

    ncclUniqueId id;
    NCCL_CHECK(ncclGetUniqueId(&id));

    std::vector<std::shared_ptr<NCCLCommunicator>> communicators(world_size);
    std::vector<std::thread> threads;
    std::mutex comm_mutex;
    std::exception_ptr init_exception;
    SimpleBarrier sync_barrier(world_size);

    for (int rank = 0; rank < world_size; ++rank) {
        threads.emplace_back([&, rank]() {
            try {
                int device_id = device_ids[rank];
                CUDA_CHECK(cudaSetDevice(device_id));

                cudaStream_t stream;
                CUDA_CHECK(cudaStreamCreate(&stream));

                ncclUniqueId local_id;
                std::memcpy(&local_id, &id, sizeof(ncclUniqueId));
                sync_barrier.arrive_and_wait();

                ncclComm_t comm;
                NCCL_CHECK(ncclCommInitRank(&comm, world_size, local_id, rank));

                auto comm_obj = std::shared_ptr<NCCLCommunicator>(
                    new NCCLCommunicator(rank, world_size, device_ids, comm, 
                                         reinterpret_cast<LlaisysCudaStream_t>(stream)));

                {
                    std::lock_guard<std::mutex> lock(comm_mutex);
                    communicators[rank] = comm_obj;
                }
            } catch (...) {
                std::lock_guard<std::mutex> lock(comm_mutex);
                if (!init_exception) {
                    init_exception = std::current_exception();
                }
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    if (init_exception) {
        std::rethrow_exception(init_exception);
    }

    return communicators;
}

NCCLCommunicator::NCCLCommunicator(int rank, int world_size, 
                                   const std::vector<int>& device_ids,
                                   ncclComm_t comm, LlaisysCudaStream_t stream)
    : device_ids_(device_ids), world_size_(world_size), rank_(rank),
      device_id_(device_ids[rank]) {
    impl_ = std::make_unique<Impl>();
    impl_->comm = comm;
    impl_->stream = stream;
}

NCCLCommunicator::~NCCLCommunicator() {
    if (impl_) {
        if (impl_->stream) {
            cudaStreamDestroy(reinterpret_cast<cudaStream_t>(impl_->stream));
        }
        if (impl_->comm) {
            ncclCommDestroy(impl_->comm);
        }
    }
}

void NCCLCommunicator::setDevice() const {
    CUDA_CHECK(cudaSetDevice(device_id_));
}

void NCCLCommunicator::allReduce(void* buff, size_t count, int dtype) {
    CUDA_CHECK(cudaSetDevice(device_id_));
    ncclDataType_t nccl_dtype = static_cast<ncclDataType_t>(toNCCLDataType(dtype));
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(impl_->stream);
    NCCL_CHECK(ncclAllReduce(buff, buff, count, nccl_dtype, ncclSum, impl_->comm, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void NCCLCommunicator::allReduce(const void* sendbuff, void* recvbuff,
                                     size_t count, int dtype) {
    ncclDataType_t nccl_dtype = static_cast<ncclDataType_t>(toNCCLDataType(dtype));
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(impl_->stream);
    NCCL_CHECK(ncclAllReduce(sendbuff, recvbuff, count, nccl_dtype, ncclSum, impl_->comm, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void NCCLCommunicator::allReduceAsync(void* buff, size_t count, int dtype) {
    ncclDataType_t nccl_dtype = static_cast<ncclDataType_t>(toNCCLDataType(dtype));
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(impl_->stream);
    NCCL_CHECK(ncclAllReduce(buff, buff, count, nccl_dtype, ncclSum, impl_->comm, stream));
}

void NCCLCommunicator::allReduceAsync(const void* sendbuff, void* recvbuff,
                                          size_t count, int dtype) {
    ncclDataType_t nccl_dtype = static_cast<ncclDataType_t>(toNCCLDataType(dtype));
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(impl_->stream);
    NCCL_CHECK(ncclAllReduce(sendbuff, recvbuff, count, nccl_dtype, ncclSum, impl_->comm, stream));
}

void NCCLCommunicator::streamSynchronize() {
    CUDA_CHECK(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(impl_->stream)));
}

void NCCLCommunicator::synchronizeAll(
    const std::vector<std::shared_ptr<NCCLCommunicator>>& comms) {
    for (const auto& comm : comms) {
        if (comm) {
            CUDA_CHECK(cudaSetDevice(comm->device_id()));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
}

// Map llaisys DataType to NCCL data type
// llaisys DataType enum (from python/llaisys/libllaisys/llaisys_types.py):
// F32 = 13, F16 = 12, BF16 = 19, I32 = 5, I64 = 6
int toNCCLDataType(int dtype) {
    switch (dtype) {
        case 13: return ncclFloat32;  // F32
        case 12: return ncclFloat16;  // F16
        case 19: return ncclBfloat16; // BF16
        case 5:  return ncclInt32;    // I32
        case 6:  return ncclInt64;    // I64
        default: throw std::invalid_argument("Unsupported data type for NCCL: " + std::to_string(dtype));
    }
}

size_t getNCCLDataTypeSize(int dtype) {
    ncclDataType_t nccl_dtype = static_cast<ncclDataType_t>(dtype);
    switch (nccl_dtype) {
        case ncclFloat32:
        case ncclInt32:
            return 4;
        case ncclFloat16:
        case ncclBfloat16:
            return 2;
        case ncclInt64:
        case ncclUint64:
        case ncclFloat64:
            return 8;
        case ncclInt8:
        case ncclUint8:
            return 1;
        default:
            throw std::invalid_argument("Unknown NCCL data type");
    }
}

} // namespace nvidia
} // namespace device
} // namespace llaisys
