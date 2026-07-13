#include "tensor.hpp"
#include <functional>
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
    size_t accumulated = 1;
    for(size_t i=0;i<ndim();i++)
    {
        size_t current_dim=ndim()-1-i;
        if(_meta.strides[current_dim]!=static_cast<ptrdiff_t>(accumulated))
        {
            return false;
        }
        accumulated*=_meta.shape[current_dim];
    }
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    if(order.size()!=ndim()){
        std::cerr << "Error: permute order size mismatch. Expected " << ndim()
                  << " but got " << order.size() << std::endl;
        return nullptr;
    }
    std::vector<size_t> new_shape(this->ndim());
    std::vector<ptrdiff_t> new_strides(this->ndim());
    
    for(size_t i=0;i<this->ndim();++i)
    {
        size_t old_idx=order[i];
        if(old_idx>=this->ndim()){
            std::cerr << "Error: permute order index out of range. Got " << old_idx << std::endl;
            return nullptr;
        }
        new_shape[i]=_meta.shape[old_idx];
        new_strides[i]=_meta.strides[old_idx];
    }
    TensorMeta new_meta{_meta.dtype,new_shape,new_strides};
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    size_t new_numel=1;
    for(auto s:shape){
        new_numel*=s;
    }
    if(new_numel!=this->numel()){
        std::cerr << "Error: view size mismatch. Expected " << this->numel()
                  << " but got " << new_numel << std::endl;
        return nullptr;
    }

    if(!this->isContiguous()){
        std::cerr << "Error: view requires contiguous tensor." << std::endl;
        return nullptr;
    }
    std::vector<ptrdiff_t> new_strides(shape.size());
size_t stride = 1;

if (!shape.empty()) {
    // 安全的 size_t 逆序循环写法
    for (size_t idx = shape.size(); idx-- > 0; ) {
    new_strides[idx] = stride;
    stride *= shape[idx];
}
}

    TensorMeta new_meta{_meta.dtype,shape,new_strides};
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
   if(dim>=this->ndim()){
       std::cerr << "Error: slice dimension out of range. Got " << dim << std::endl;
       return nullptr;
    }
    if(start>=end||end>this->shape()[dim]){
        std::cerr << "Error: slice indices out of range. Got [" << start << ", " << end << ")"
                  << " for dimension size " << this->shape()[dim] << std::endl;
        return nullptr;
    }

    std::vector<size_t> new_shape=this->shape();
    new_shape[dim]=end-start;

    std::vector<ptrdiff_t> new_strides=this->strides();

    size_t skipped_elements=start*new_strides[dim];
    size_t offset_bytes=skipped_elements*this->elementSize();
    size_t new_offset=this->_offset+offset_bytes;

    TensorMeta new_meta{this->dtype(),new_shape,new_strides};
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, new_offset));
}

void Tensor::load(const void *src_) {
    size_t size=this->numel()*this->elementSize();
    void *dst=this->data();
    llaisysMemcpyKind_t kind=LLAISYS_MEMCPY_H2H;
    if(this->deviceType()==LLAISYS_DEVICE_NVIDIA){
        kind=LLAISYS_MEMCPY_H2D;
    }
    core::context().runtime().api()->memcpy_sync(dst,src_,size,kind);
}

tensor_t Tensor::contiguous() const {
    if (this->isContiguous()) {
        return std::shared_ptr<Tensor>(new Tensor(_meta, _storage, _offset));
    }
    auto res = Tensor::create(
        this->shape(), this->dtype(), this->deviceType(), this->deviceId()
    );

    char* dst_ptr = reinterpret_cast<char*>(res->data());
    const char* src_base = reinterpret_cast<const char*>(this->data());
    
    size_t elem_size = this->elementSize();    
    const auto& shape = this->shape();
    const auto& strides = this->strides();
    size_t ndim = this->ndim();

    std::function<void(size_t, size_t)> recursive_copy = 
        [&](size_t dim, size_t src_offset) {
            
        if (ndim == 0) {
            std::memcpy(dst_ptr, src_base, elem_size);
            dst_ptr += elem_size;
            return;
        }

        if (dim == ndim - 1) {
            size_t stride_bytes = strides[dim] * elem_size;
            for (size_t i = 0; i < shape[dim]; ++i) {
                std::memcpy(dst_ptr, src_base + src_offset + i * stride_bytes, elem_size);
                dst_ptr += elem_size; 
            }
        } else {
            size_t stride_bytes = strides[dim] * elem_size;
            for (size_t i = 0; i < shape[dim]; ++i) {
                recursive_copy(dim + 1, src_offset + i * stride_bytes);
            }
        }
    };

    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        recursive_copy(0, 0);
    } else {
        std::cerr << "Error: contiguous for GPU is not implemented." << std::endl;
    }

    return res;
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    size_t new_numel = 1;
    for (auto s : shape) new_numel *= s;
    
    if (new_numel != this->numel()) {
        std::cerr << "Error: reshape size mismatch." << std::endl;
        return nullptr;
    }

    
    if (this->isContiguous()) {
        return this->view(shape);
    } else {
        return this->contiguous()->view(shape);
    }
    
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    if (device_type == this->deviceType() && device == this->deviceId()) {
        return std::shared_ptr<Tensor>(new Tensor(_meta, _storage, _offset));
    }

    tensor_t src_contiguous = this->contiguous();


    auto res = Tensor::create(src_contiguous->shape(), src_contiguous->dtype(), device_type, device);

    void *dst_ptr = res->data();
    const void *src_ptr = src_contiguous->data();
    size_t size = src_contiguous->numel() * src_contiguous->elementSize();

    llaisysMemcpyKind_t kind;

    bool is_src_gpu = (this->deviceType() == LLAISYS_DEVICE_NVIDIA);
    bool is_dst_gpu = (device_type == LLAISYS_DEVICE_NVIDIA);

    if (!is_src_gpu && is_dst_gpu) {
        kind = LLAISYS_MEMCPY_H2D; // Host -> Device (CPU to GPU)
    } else if (is_src_gpu && !is_dst_gpu) {
        kind = LLAISYS_MEMCPY_D2H; // Device -> Host (GPU to CPU)
    } else if (is_src_gpu && is_dst_gpu) {
        kind = LLAISYS_MEMCPY_D2D; // Device -> Device (GPU to GPU)
    } else {
        kind = LLAISYS_MEMCPY_H2H; // Host -> Host (CPU to CPU)
    }

    core::context().runtime().api()->memcpy_sync(dst_ptr, src_ptr, size, kind);

    return res;
}

} // namespace llaisys
// Force recompile
