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
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
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
    // 空张量或单元素张量总是连续的
    if (_meta.shape.empty() || numel() <= 1) {
        return true;
    }
    
    // 计算预期的连续步长（行优先顺序）
    std::vector<ptrdiff_t> expected_strides(_meta.shape.size());
    ptrdiff_t stride = 1;
    for (int i = static_cast<int>(_meta.shape.size()) - 1; i >= 0; i--) {
        expected_strides[i] = stride;
        stride *= _meta.shape[i];
    }
    
    // 比较实际步长与期望步长
    return _meta.strides == expected_strides;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    // 检查order的有效性
    ASSERT(order.size() == _meta.shape.size(), "Permute: order size must match tensor dimensions");
    
    std::vector<bool> used(order.size(), false);
    for (size_t i = 0; i < order.size(); i++) {
        ASSERT(order[i] < order.size(), "Permute: invalid dimension index");
        ASSERT(!used[order[i]], "Permute: duplicate dimension index");
        used[order[i]] = true;
    }
    
    // 根据order重新排列形状和步长
    TensorMeta new_meta;
    new_meta.dtype = _meta.dtype;
    new_meta.shape.resize(order.size());
    new_meta.strides.resize(order.size());
    
    for (size_t i = 0; i < order.size(); i++) {
        new_meta.shape[i] = _meta.shape[order[i]];
        new_meta.strides[i] = _meta.strides[order[i]];
    }
    
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    // 计算新形状的元素总数
    size_t new_numel = 1;
    for (size_t dim : shape) {
        new_numel *= dim;
    }
    
    // 检查元素总数是否一致
    ASSERT(new_numel == this->numel(), "View: total number of elements must be preserved");
    
    // 检查原张量是否连续，非连续张量不能直接view
    ASSERT(this->isContiguous(), "View: tensor must be contiguous");
    
    // 为新形状计算连续步长
    TensorMeta new_meta;
    new_meta.dtype = _meta.dtype;
    new_meta.shape = shape;
    new_meta.strides.resize(shape.size());
    
    if (!shape.empty()) {
        ptrdiff_t stride = 1;
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; i--) {
            new_meta.strides[i] = stride;
            stride *= shape[i];
        }
    }
    
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    // 检查维度索引有效性
    ASSERT(dim < _meta.shape.size(), "Slice: dimension index out of bounds");
    
    // 检查切片范围有效性
    ASSERT(start <= end, "Slice: start index must be <= end index");
    ASSERT(end <= _meta.shape[dim], "Slice: end index out of bounds");
    
    // 计算新的形状
    TensorMeta new_meta = _meta;  // 复制原始元数据
    new_meta.shape[dim] = end - start;  // 只更新被切片维度的大小
    
    // 计算新的偏移量
    size_t new_offset = _offset + start * _meta.strides[dim] * elementSize();
    
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, new_offset));
}

void Tensor::load(const void *src_) {
    core::context().setDevice(this->deviceType(), this->deviceId());
    size_t total_bytes = this->numel() * this->elementSize();
    
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        // 对于CPU设备，直接使用内存复制
        core::context().runtime().api()->memcpy_sync(
            this->data(),
            src_,
            total_bytes,
            LLAISYS_MEMCPY_H2H);
    } else {
        // 对于其他设备，使用主机到设备的内存复制
        core::context().runtime().api()->memcpy_sync(
            this->data(),
            src_,
            total_bytes,
            LLAISYS_MEMCPY_H2D);
    }
}

tensor_t Tensor::contiguous() const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

} // namespace llaisys
