#pragma once

#include <vector>
#include <memory>
#include <stdexcept>
#include <string>

// Forward declare NCCL types
typedef struct ncclComm* ncclComm_t;

// Forward declare CUDA stream type (it's a pointer type in CUDA)
// We use a different name to avoid conflicts
typedef void* LlaisysCudaStream_t;

namespace llaisys {
namespace device {
namespace nvidia {

// NCCL data type enum (matches NCCL definitions)
enum class NCCLDataType {
    Float32 = 0,
    Float64 = 1,
    Float16 = 2,
    Bfloat16 = 3,
    Int8 = 4,
    Int32 = 5,
    Int64 = 6,
    Uint8 = 7,
    Uint32 = 8,
    Uint64 = 9
};

// NCCL Communicator for tensor parallel operations
// This class manages NCCL communication across multiple GPUs
class NCCLCommunicator {
public:
    // Create communicators for a group of GPUs (one per rank)
    // This should be called once to create all communicators
    static std::vector<std::shared_ptr<NCCLCommunicator>> createAll(
        const std::vector<int>& device_ids);

    ~NCCLCommunicator();

    // Disable copy
    NCCLCommunicator(const NCCLCommunicator&) = delete;
    NCCLCommunicator& operator=(const NCCLCommunicator&) = delete;

    // Getters
    int world_size() const { return world_size_; }
    int rank() const { return rank_; }
    int device_id() const { return device_id_; }
    const std::vector<int>& device_ids() const { return device_ids_; }

    // Get internal handles (for CUDA code)
    ncclComm_t comm() const;
    LlaisysCudaStream_t stream() const;

    // Set device for current thread
    void setDevice() const;

    // All-Reduce: sum operation
    void allReduce(void* buff, size_t count, int dtype);
    void allReduce(const void* sendbuff, void* recvbuff, size_t count, int dtype);

    // All-Reduce async
    void allReduceAsync(void* buff, size_t count, int dtype);
    void allReduceAsync(const void* sendbuff, void* recvbuff, size_t count, int dtype);

    // Synchronize stream
    void streamSynchronize();

    // Synchronize all devices in the group
    static void synchronizeAll(const std::vector<std::shared_ptr<NCCLCommunicator>>& comms);

public:
    // Constructor for direct creation (used by worker threads)
    NCCLCommunicator(int rank, int world_size, const std::vector<int>& device_ids,
                     ncclComm_t comm, LlaisysCudaStream_t stream);
    
private:

    // Allow implementation to access private members
    class Impl;
    friend class Impl;
    friend class NCCLCommunicatorImpl;

    std::vector<int> device_ids_;
    int world_size_;
    int rank_;
    int device_id_;
    std::unique_ptr<Impl> impl_;
};

// Helper to convert data type to NCCL type
int toNCCLDataType(int dtype);

// Get element size for NCCL data type
size_t getNCCLDataTypeSize(int dtype);

} // namespace nvidia
} // namespace device
} // namespace llaisys
