#include "nccl_context.hpp"
#include "cuda_utils.cuh"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>

namespace llaisys::device::nvidia {
namespace {

std::string sanitizeForPath(const std::string &value) {
    std::string sanitized = value;
    for (char &ch : sanitized) {
        const bool keep = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '-' || ch == '_';
        if (!keep) {
            ch = '_';
        }
    }
    return sanitized;
}

std::string envOrEmpty(const char *name) {
    const char *value = std::getenv(name);
    if (value == nullptr) {
        return {};
    }
    return value;
}

} // namespace

ncclDataType_t toNcclDataType(llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BOOL:
        return ncclUint8;
    case LLAISYS_DTYPE_I8:
        return ncclInt8;
    case LLAISYS_DTYPE_I32:
        return ncclInt32;
    case LLAISYS_DTYPE_I64:
        return ncclInt64;
    case LLAISYS_DTYPE_U8:
    case LLAISYS_DTYPE_BYTE:
        return ncclUint8;
    case LLAISYS_DTYPE_U32:
        return ncclUint32;
    case LLAISYS_DTYPE_U64:
        return ncclUint64;
    case LLAISYS_DTYPE_F16:
        return ncclFloat16;
    case LLAISYS_DTYPE_F32:
        return ncclFloat32;
    case LLAISYS_DTYPE_F64:
        return ncclFloat64;
    case LLAISYS_DTYPE_BF16:
        return ncclBfloat16;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

ncclRedOp_t toNcclReduceOp(NcclReduceOp op) {
    switch (op) {
    case NcclReduceOp::Sum:
        return ncclSum;
    default:
        throw std::invalid_argument("Unsupported NCCL reduce op");
    }
}

std::string resolveBootstrapPath() {
    const std::string explicit_path = envOrEmpty("LLAISYS_DIST_BOOTSTRAP_PATH");
    if (!explicit_path.empty()) {
        return explicit_path;
    }

    const std::string dist_id = envOrEmpty("LLAISYS_DIST_ID");
    if (!dist_id.empty()) {
        return "/tmp/llaisys_nccl_" + sanitizeForPath(dist_id) + ".bin";
    }

    const std::string master_addr = envOrEmpty("MASTER_ADDR");
    const std::string master_port = envOrEmpty("MASTER_PORT");
    if (!master_addr.empty() && !master_port.empty()) {
        return "/tmp/llaisys_nccl_" + sanitizeForPath(master_addr) + "_" + sanitizeForPath(master_port) + ".bin";
    }

    throw std::invalid_argument(
        "Distributed NCCL init requires LLAISYS_DIST_BOOTSTRAP_PATH, LLAISYS_DIST_ID, or MASTER_ADDR/MASTER_PORT.");
}

void storeUniqueId(const std::string &path, const ncclUniqueId &unique_id) {
    const std::filesystem::path file_path(path);
    const std::filesystem::path dir_path = file_path.parent_path();
    if (!dir_path.empty()) {
        std::filesystem::create_directories(dir_path);
    }

    std::error_code remove_ec;
    std::filesystem::remove(file_path, remove_ec);

    std::ostringstream temp_path_builder;
    temp_path_builder << path << ".tmp." << ::getpid();
    const std::string temp_path = temp_path_builder.str();

    {
        std::ofstream output(temp_path, std::ios::binary | std::ios::trunc);
        CHECK_ARGUMENT(output.good(), "failed to open NCCL bootstrap temp file");
        output.write(reinterpret_cast<const char *>(&unique_id), sizeof(unique_id));
        CHECK_ARGUMENT(output.good(), "failed to write NCCL bootstrap data");
        output.flush();
        CHECK_ARGUMENT(output.good(), "failed to flush NCCL bootstrap data");
    }

    std::filesystem::rename(temp_path, path);
}

ncclUniqueId loadUniqueId(int rank) {
    const std::string path = resolveBootstrapPath();
    ncclUniqueId unique_id{};
    if (rank == 0) {
        NCCL_CHECK(ncclGetUniqueId(&unique_id));
        storeUniqueId(path, unique_id);
        return unique_id;
    }

    constexpr int kMaxAttempts = 600;
    constexpr auto kPollInterval = std::chrono::milliseconds(100);
    constexpr auto kFreshWindow = std::chrono::seconds(5);
    for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
        std::error_code ec;
        const bool exists = std::filesystem::exists(path, ec);
        const bool size_ok = exists && !ec && std::filesystem::file_size(path, ec) == sizeof(unique_id);
        const auto min_write_time = std::filesystem::file_time_type::clock::now() - kFreshWindow;
        const bool fresh_enough = size_ok && !ec && std::filesystem::last_write_time(path, ec) >= min_write_time;
        if (fresh_enough && !ec) {
            std::ifstream input(path, std::ios::binary);
            if (input.good()) {
                input.read(reinterpret_cast<char *>(&unique_id), sizeof(unique_id));
                if (input.good()) {
                    return unique_id;
                }
            }
        }
        std::this_thread::sleep_for(kPollInterval);
    }

    throw std::runtime_error("Timed out waiting for NCCL bootstrap file");
}

NcclContext::NcclContext(int rank, int world_size, int device_id)
    : _rank(rank),
      _world_size(world_size),
      _device_id(device_id),
      _comm(nullptr),
      _barrier_buffer(nullptr) {
    CHECK_ARGUMENT(world_size > 0, "world_size must be positive");
    CHECK_ARGUMENT(rank >= 0 && rank < world_size, "rank must be in [0, world_size)");
    CHECK_ARGUMENT(device_id >= 0, "device_id must be non-negative");

    CUDA_CHECK(cudaSetDevice(device_id));
    const ncclUniqueId unique_id = loadUniqueId(rank);
    NCCL_CHECK(ncclCommInitRank(&_comm, world_size, unique_id, rank));
    CUDA_CHECK(cudaMalloc(&_barrier_buffer, sizeof(int32_t)));
}

NcclContext::~NcclContext() {
    if (_barrier_buffer != nullptr) {
        cudaFree(_barrier_buffer);
        _barrier_buffer = nullptr;
    }
    if (_comm != nullptr) {
        ncclCommDestroy(_comm);
        _comm = nullptr;
    }
}

int NcclContext::rank() const {
    return _rank;
}

int NcclContext::worldSize() const {
    return _world_size;
}

int NcclContext::deviceId() const {
    return _device_id;
}

void NcclContext::allReduce(void *data, size_t count, llaisysDataType_t dtype, llaisysStream_t stream, NcclReduceOp op) const {
    CHECK_ARGUMENT(data != nullptr || count == 0, "allReduce data must not be null");
    std::lock_guard<std::mutex> lock(_mutex);
    NCCL_CHECK(ncclAllReduce(data, data, count, toNcclDataType(dtype), toNcclReduceOp(op), _comm, toCudaStream(stream)));
}

void NcclContext::allGather(const void *send_buffer, void *recv_buffer, size_t count, llaisysDataType_t dtype, llaisysStream_t stream) const {
    CHECK_ARGUMENT(send_buffer != nullptr || count == 0, "allGather send buffer must not be null");
    CHECK_ARGUMENT(recv_buffer != nullptr || count == 0, "allGather recv buffer must not be null");
    std::lock_guard<std::mutex> lock(_mutex);
    NCCL_CHECK(ncclAllGather(send_buffer, recv_buffer, count, toNcclDataType(dtype), _comm, toCudaStream(stream)));
}

void NcclContext::broadcast(void *data, size_t count, llaisysDataType_t dtype, int root, llaisysStream_t stream) const {
    CHECK_ARGUMENT(data != nullptr || count == 0, "broadcast data must not be null");
    CHECK_ARGUMENT(root >= 0 && root < _world_size, "broadcast root must be in [0, world_size)");
    std::lock_guard<std::mutex> lock(_mutex);
    NCCL_CHECK(ncclBroadcast(data, data, count, toNcclDataType(dtype), root, _comm, toCudaStream(stream)));
}

void NcclContext::barrier(llaisysStream_t stream) const {
    std::lock_guard<std::mutex> lock(_mutex);
    CUDA_CHECK(cudaMemsetAsync(_barrier_buffer, 0, sizeof(int32_t), toCudaStream(stream)));
    NCCL_CHECK(ncclAllReduce(_barrier_buffer, _barrier_buffer, 1, ncclInt32, ncclSum, _comm, toCudaStream(stream)));
}

} // namespace llaisys::device::nvidia
