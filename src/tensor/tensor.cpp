#include "tensor.hpp"

#include "../utils.hpp"

#include <cstddef>
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
    size_t ndim_ = this->ndim();
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        if (this->_meta.strides[ndim_ - i] != stride) {
            return false;
        }
        stride *= this->_meta.shape[ndim_ - i];
    }
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    size_t ndim_ = this->ndim();
    CHECK_ARGUMENT(order.size() == ndim_, "order size must be equal to ndim");
    // 检查order是否包含所有维度
    std::vector<bool> used(ndim_, false);
    for (auto idx : order) {
        if (idx >= ndim_ || used[idx]) {
            CHECK_ARGUMENT(false, "Invalid permutation order");
        }
        used[idx] = true;
    }

    std::vector<size_t> shape(ndim_);
    std::vector<ptrdiff_t> strides(ndim_);
    for (size_t i = 0; i < ndim_; i++) {
        shape[i] = this->_meta.shape[order[i]];
        strides[i] = this->_meta.strides[order[i]];
    }
    TensorMeta meta{this->_meta.dtype, shape, strides};
    return std::shared_ptr<Tensor>(new Tensor(meta, _storage, _offset));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    // 计算新形状的元素总数
    size_t new_numel = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
    // 检查元素总数是否与原张量相同
    if (new_numel != this->numel()) {
        CHECK_ARGUMENT(false, "New shape has different number of elements");
    }
    
    // 对于连续张量，直接计算新的步长
    if(isContiguous()) {
        size_t ndim_ = shape.size();
        std::vector<ptrdiff_t> strides(ndim_);
        size_t stride = 1;
        for (size_t i = 1; i <= ndim_; i++) {
            strides[ndim_ - i] = stride;
            stride *= shape[ndim_ - i];
        }
        TensorMeta meta{this->_meta.dtype, shape, strides};
        return std::shared_ptr<Tensor>(new Tensor(meta, _storage, _offset));
    } else {
        // 对于非连续张量，检查是否可以进行视图操作
        // 这里简化实现，实际中可能需要更复杂的检查
        CHECK_ARGUMENT(false, "Cannot view non-contiguous tensor");
    }
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    // 检查dim是否有效
    if (dim >= this->ndim()) {
        CHECK_ARGUMENT(false, "Dimension out of range");
    }
    
    // 检查start和end是否有效
    if (start >= end || end > this->_meta.shape[dim]) {
        CHECK_ARGUMENT(false, "Invalid start or end indices");
    }
    
    // 创建新的形状
    std::vector<size_t> new_shape = this->_meta.shape;
    new_shape[dim] = end - start;
    
    // 计算新的偏移量
    size_t new_offset = this->_offset;
    new_offset += start * this->_meta.strides[dim] * this->elementSize();
    
    // 创建新的步长（保持不变）
    std::vector<ptrdiff_t> new_strides = this->_meta.strides;
    
    // 创建新的meta
    TensorMeta meta{this->_meta.dtype, new_shape, new_strides};
    
    // 创建并返回新的张量
    return std::shared_ptr<Tensor>(new Tensor(meta, _storage, new_offset));
}

void Tensor::load(const void *src_) {
    // 计算需要复制的字节数
    size_t bytes_to_copy = this->numel() * this->elementSize();
    
    // 检查存储大小是否足够
    if (this->_storage->size() < this->_offset + bytes_to_copy) {
        CHECK_ARGUMENT(false, "Storage size is not sufficient");
    }
    
    // 获取目标内存地址（考虑偏移量）
    std::byte *dst = this->data();
    
    // 根据设备类型选择合适的内存复制方式
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        // CPU到CPU的复制
        std::memcpy(dst, src_, bytes_to_copy);
    } else {
        // 主机到设备的复制
        core::context().setDevice(this->deviceType(), this->deviceId());
        core::context().runtime().api()->memcpy_sync(
            dst,
            src_,
            bytes_to_copy,
            LLAISYS_MEMCPY_H2D
        );
        // 同步设备确保复制完成
        core::context().runtime().api()->device_synchronize();
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
