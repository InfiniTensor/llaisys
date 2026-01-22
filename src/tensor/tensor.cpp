#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);

    // Calculate strides
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    // 当请求 CPU 内存但当前运行时环境不是 CPU 时（比如 GPU 环境）
    // 分配 可被 GPU 访问的主机内存
    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        // 对于 GPU 请求：分配 GPU 内存
        // 对于纯 CPU 环境：分配普通 CPU 内存
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

// 维度
size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

// 元素总数
size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

bool Tensor::isContiguous() const {
    // A tensor is contiguous when its strides match the standard C-order dense layout.
    const auto &sh = this->shape();
    const auto &st = this->strides();
    size_t n = sh.size();
    if (n == 0) {
        return true;
    }
    size_t expected = 1;
    for (ptrdiff_t i = static_cast<ptrdiff_t>(n) - 1; i >= 0; --i) {
        if (static_cast<size_t>(st[i]) != expected) {
            return false;
        }
        expected *= sh[i];
    }
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    if (order.size() != this->ndim()) {
        throw std::invalid_argument("permute: order size must equal number of dimensions");
    }
    size_t n = this->ndim();
    std::vector<char> seen(n, 0);
    for (size_t v : order) {
        if (v >= n) {
            throw std::out_of_range("permute: index out of range");
        }
        if (seen[v]) {
            throw std::invalid_argument("permute: duplicate dimension in order");
        }
        seen[v] = 1;
    }

    TensorMeta meta;
    meta.dtype = this->dtype();
    meta.shape.resize(n);
    meta.strides.resize(n);
    for (size_t i = 0; i < n; ++i) {
        meta.shape[i] = _meta.shape[order[i]];
        meta.strides[i] = _meta.strides[order[i]];
    }

    return std::shared_ptr<Tensor>(new Tensor(std::move(meta), _storage, _offset));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    // view is only allowed when number of elements match and tensor is contiguous
    size_t new_numel = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
    if (new_numel != this->numel()) {
        throw std::invalid_argument("view: total elements mismatch");
    }
    if (!this->isContiguous()) {
        throw std::runtime_error("view: tensor must be contiguous to view");
    }

    // compute new contiguous strides
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> new_strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; ++i) {
        new_strides[ndim_ - i] = static_cast<ptrdiff_t>(stride);
        stride *= shape[ndim_ - i];
    }

    TensorMeta meta{this->dtype(), shape, new_strides};
    return std::shared_ptr<Tensor>(new Tensor(std::move(meta), _storage, _offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    if (dim >= this->ndim()) {
        throw std::out_of_range("slice: dim out of range");
    }
    if (start > end || end > _meta.shape[dim]) {
        throw std::out_of_range("slice: invalid start/end");
    }

    TensorMeta meta = _meta;
    meta.shape[dim] = end - start;

    // offset increases by start * stride[dim] elements -> bytes
    size_t byte_offset = static_cast<size_t>(start * _meta.strides[dim]) * this->elementSize();
    return std::shared_ptr<Tensor>(new Tensor(std::move(meta), _storage, _offset + byte_offset));
}

void Tensor::load(const void *src_) {
    if (src_ == nullptr) {
        return;
    }

    // Ensure runtime/context points to this tensor's device
    core::context().setDevice(this->deviceType(), this->deviceId());

    // total bytes to copy
    size_t bytes = this->numel() * this->elementSize();

    // If storage is host-accessible, do a direct memcpy.
    // Otherwise perform a host->device memcpy via the runtime API.
    if (_storage->isHost()) {
        std::memcpy(this->data(), src_, bytes);
    } else {
        core::context().runtime().api()->memcpy_sync(
            this->data(),
            src_,
            bytes,
            LLAISYS_MEMCPY_H2D);
    }
}

tensor_t Tensor::contiguous() const {
    if (this->isContiguous()) {
        // returning a new Tensor object that shares the same storage/offset/meta
        return std::shared_ptr<Tensor>(new Tensor(_meta, _storage, _offset));
    }

    // For now we implement contiguous() for host-accessible tensors.
    if (!_storage->isHost()) {
        throw std::runtime_error("contiguous: not implemented for non-host storage");
    }

    // allocate destination contiguous tensor on the same device/type
    auto dst = Tensor::create(this->shape(), this->dtype(), this->deviceType(), this->deviceId());

    size_t elem_sz = this->elementSize();
    // copy element-by-element using strides
    size_t n = this->numel();
    const std::vector<size_t> &sh = this->shape();
    const std::vector<ptrdiff_t> &st = this->strides();
    std::vector<size_t> idx(sh.size(), 0);

    for (size_t linear = 0; linear < n; ++linear) {
        // compute source element index in elements (not bytes)
        size_t rem = linear;
        size_t src_elem_index = 0;
        for (size_t d = 0; d < sh.size(); ++d) {
            // compute multi-index for dimension d
            size_t dim_idx = 0;
            // We can compute multi-index by dividing rem by product of later dims
            // Precompute multipliers could be faster, but compute here directly.
            size_t mult = 1;
            for (size_t k = d + 1; k < sh.size(); ++k) {
                mult *= sh[k];
            }
            if (mult != 0) {
                dim_idx = rem / mult;
                rem = rem % mult;
            } else {
                dim_idx = 0;
            }
            src_elem_index += static_cast<size_t>(st[d]) * dim_idx;
        }
        const std::byte *src_ptr = this->data() + src_elem_index * elem_sz;
        std::byte *dst_ptr = dst->data() + linear * elem_sz;
        std::memcpy(dst_ptr, src_ptr, elem_sz);
    }

    return dst;
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    return this->view(shape);
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    // If destination is same as current, return a tensor that shares storage (no copy).
    if (device_type == this->deviceType() && (device < 0 || device == this->deviceId())) {
        return std::shared_ptr<Tensor>(new Tensor(_meta, _storage, _offset));
    }

    size_t bytes = this->numel() * this->elementSize();

    // Set context to source device so that create(...) for CPU allocations uses the correct runtime when needed.
    core::context().setDevice(this->deviceType(), this->deviceId());

    // create destination tensor on requested device
    auto dst = Tensor::create(this->shape(), this->dtype(), device_type, device);

    // If both storages are host, plain memcpy
    if (this->_storage->isHost() && dst->_storage->isHost()) {
        std::memcpy(dst->data(), this->data(), bytes);
        return dst;
    }

    // If same runtime (same device type & device id), do device-to-device copy directly
    if (this->deviceType() == device_type && this->deviceId() == device) {
        core::context().runtime().api()->memcpy_sync(dst->data(), this->data(), bytes, LLAISYS_MEMCPY_D2D);
        return dst;
    }

    // Common cases:
    // - src is CPU, dst is device: H2D (current context is src runtime (CPU) but create for device may have switched context
    //   so set context back to src device to perform the appropriate memcpy)
    // - src is device, dst is CPU: D2H
    // - cross-device (e.g., GPU_A -> GPU_B): use host staging: D2H from src runtime, then H2D to dst runtime.

    // Case: src is CPU, dst is device
    if (this->deviceType() == LLAISYS_DEVICE_CPU && device_type != LLAISYS_DEVICE_CPU) {
        // set context to dst device runtime (create may have changed it already, but ensure)
        core::context().setDevice(device_type, device);
        core::context().runtime().api()->memcpy_sync(dst->data(), this->data(), bytes, LLAISYS_MEMCPY_H2D);
        return dst;
    }

    // Case: src is device, dst is CPU
    if (this->deviceType() != LLAISYS_DEVICE_CPU && device_type == LLAISYS_DEVICE_CPU) {
        // set context to src device
        core::context().setDevice(this->deviceType(), this->deviceId());
        core::context().runtime().api()->memcpy_sync(dst->data(), this->data(), bytes, LLAISYS_MEMCPY_D2H);
        return dst;
    }

    // Case: cross-device (both non-CPU and different devices / runtimes)
    // Use host staging:
    auto host_tmp = Tensor::create(this->shape(), this->dtype(), LLAISYS_DEVICE_CPU);
    // copy src -> host_tmp (set context to src)
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->memcpy_sync(host_tmp->data(), this->data(), bytes, LLAISYS_MEMCPY_D2H);
    // copy host_tmp -> dst (set context to dst)
    core::context().setDevice(device_type, device);
    core::context().runtime().api()->memcpy_sync(dst->data(), host_tmp->data(), bytes, LLAISYS_MEMCPY_H2D);

    return dst;
}

} // namespace llaisys
