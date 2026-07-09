#include "tensor.hpp"
#include "../../include/llaisys.h"
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
    auto shape = this->shape();    
    auto strides = this->strides();    
    if (shape.empty()) {    
        return true;   
    }    
      
    for (size_t i = 0; i < shape.size(); ++i) {    
        size_t expected_stride = 1;    
        for (size_t j = i + 1; j < shape.size(); ++j) {    
            expected_stride *= shape[j];    
        }    
          
        if (static_cast<size_t>(strides[i]) != expected_stride) {    
            return false;    
        }    
    }    
    return true;  
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    if (order.size() != this->shape().size()) {  
        throw std::invalid_argument("permute order must match tensor dimensions");  
    }  
    
    std::vector<bool> used(this->shape().size(), false);  
    for (size_t idx : order) {  
        if (idx >= this->shape().size()) {  
            throw std::invalid_argument("permute order contains invalid dimension index");  
        }  
        if (used[idx]) {  
            throw std::invalid_argument("permute order contains duplicate dimension indices");  
        }  
        used[idx] = true;  
    }  
       
    std::vector<size_t> new_shape;  
    std::vector<ptrdiff_t> new_strides;  
      
    for (size_t idx : order) {  
        new_shape.push_back(this->shape()[idx]);  
        new_strides.push_back(this->strides()[idx]);  
    }  
       
    TensorMeta new_meta;  
    new_meta.dtype = this->dtype();  
    new_meta.shape = new_shape;  
    new_meta.strides = new_strides;  
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, this->_offset));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    size_t new_numel = 1;  
    for (size_t dim : shape) {  
        new_numel *= dim;  
    }  
      
    if (new_numel != this->numel()) {  
        throw std::invalid_argument("view shape has different number of elements than tensor");  
    }  
      
     
    if (!this->isContiguous()) {  
        throw std::invalid_argument("view can only be applied to contiguous tensors");  
    }  
      
   
    std::vector<ptrdiff_t> new_strides(shape.size());  
    new_strides[shape.size() - 1] = 1;  
      
    for (int i = static_cast<int> (shape.size()) - 2; i >= 0; --i) {  
        new_strides[i] = new_strides[i + 1] * shape[i + 1];  
    }  
      
    
    TensorMeta new_meta;  
    new_meta.dtype = this->dtype();  
    new_meta.shape = shape;  
    new_meta.strides = new_strides;  
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, this->_offset)); 
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    if (dim >= this->shape().size()) {  
        throw std::invalid_argument("slice dimension out of range");  
    }  
    if (start >= end || end > this->shape()[dim]) {  
        throw std::invalid_argument("slice start/end indices out of range");  
    }  

    auto shape = this->shape();
    shape[dim] = end - start;
    auto new_strides = this->strides();

    TensorMeta new_meta;  
    new_meta.dtype = this->dtype();  
    new_meta.shape = shape;  
    new_meta.strides = new_strides;  
    auto _offset = this->_offset + start * new_strides[dim]*this->elementSize();

    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset)); 
}

void Tensor::load(const void *src_) {
    core::context().setDevice(deviceType(), deviceId());  
    auto &runtime = core::context().runtime();  
    runtime.api()->memcpy_sync(data(), src_, this->numel() * this->elementSize(), LLAISYS_MEMCPY_H2D); 
}


tensor_t Tensor::contiguous() const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage, this->_offset));
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    // Check contiguous and shape compatibility
    if (!this->isContiguous()) {
        throw std::runtime_error("Reshape failed: tensor must be contiguous");
    }    
    size_t new_numel = 1;
    for (auto d : shape) new_numel *= d;
    if (this->numel() != new_numel) {
        throw std::runtime_error("Reshape failed: total number of elements must not change");
    }

    // Calculate new strides
    std::vector<ptrdiff_t> new_strides(shape.size());
    if (!shape.empty()) {
        new_strides.back() = 1;
        for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
            new_strides[i] = new_strides[i + 1] * shape[i + 1];
        }
    }

    // Construct new tensor meta
    TensorMeta new_meta = this->_meta;
    new_meta.shape = std::move(shape);
    new_meta.strides = std::move(new_strides);
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, this->_offset));
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {  
    // If already on the target device, return a copy (or the same tensor)
    if (_storage->deviceType() == device_type && (device == -1 || _storage->deviceId() == device)) {
        return std::shared_ptr<Tensor>(new Tensor(_meta, _storage, this->_offset));
    }
    
    // Create a new tensor on the target device
    if (device == -1) {
        device = 0; // Default device ID
    }
    
    auto new_tensor = Tensor::create(_meta.shape, _meta.dtype, device_type, device);
    
    // Make sure source tensor is contiguous before copying
    tensor_t source = isContiguous() ? 
        std::shared_ptr<Tensor>(new Tensor(_meta, _storage, this->_offset)) : 
        contiguous();
    
    // Copy data from source device to target device
    void* src_ptr = source->data();
    void* dst_ptr = new_tensor->data();
    size_t num_bytes = source->numel() * source->elementSize();
    
    
    // Determine memcpy kind based on source and destination devices
    llaisysMemcpyKind_t kind;
    if (_storage->deviceType() == LLAISYS_DEVICE_CPU && device_type == LLAISYS_DEVICE_CPU) {
        kind = LLAISYS_MEMCPY_H2H;
    } else if (_storage->deviceType() == LLAISYS_DEVICE_CPU && device_type != LLAISYS_DEVICE_CPU) {
        kind = LLAISYS_MEMCPY_H2D;
    } else if (_storage->deviceType() != LLAISYS_DEVICE_CPU && device_type == LLAISYS_DEVICE_CPU) {
        kind = LLAISYS_MEMCPY_D2H;
    } else {
        kind = LLAISYS_MEMCPY_D2D;
    }

    llaisysDeviceType_t runtime_device_type = LLAISYS_DEVICE_CPU;
    int runtime_device_id = 0;
    if (_storage->deviceType() != LLAISYS_DEVICE_CPU) {
        runtime_device_type = _storage->deviceType();
        runtime_device_id = _storage->deviceId();
    } else if (device_type != LLAISYS_DEVICE_CPU) {
        runtime_device_type = device_type;
        runtime_device_id = device;
    }

    core::context().setDevice(runtime_device_type, runtime_device_id);
    core::context().runtime().api()->memcpy_sync(dst_ptr, src_ptr, num_bytes, kind); 
    return new_tensor;
}

} // namespace llaisys
