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
    size_t Rank=this->ndim();
    const auto&Cur_strides=this->strides();
    const auto&Shapes=this->shape();
    ptrdiff_t accumulate_stride=1;
    if(Rank==0) return true;
    for(size_t i=Rank;i>0;--i){
        size_t index=i-1;
        if(accumulate_stride!=Cur_strides[index]) return false;
        accumulate_stride*=Shapes[index];
    }
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    if(order.size()!=this->ndim()){
        throw std::runtime_error("Order Error!");
    }
    const auto&old_shape=this->shape();
    const auto&old_strides=this->strides();
    std::vector<size_t> new_shape(old_shape.size());
    std::vector<ptrdiff_t> new_strides(old_strides.size());
    for(size_t i=0;i<order.size();i++){
        size_t order_index=order[i];
        if(order_index>=this->ndim()){
            throw std::runtime_error("Index Error!");
        }
        new_shape[i]=old_shape[order_index];
        new_strides[i]=old_strides[order_index];
    }
    TensorMeta _meta{this->dtype(),std::move(new_shape),std::move(new_strides)};
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage,this->_offset));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    auto target_numel=std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
    if(this->numel()!=target_numel) throw std::runtime_error("size error");
    if (this->isContiguous()) {
        std::vector<ptrdiff_t> new_strides(shape.size());
        size_t stride = 1;
        for (long i = shape.size() - 1; i >= 0; --i) {
            new_strides[i] = stride;
            stride *= shape[i];
        }
        TensorMeta meta{this->dtype(), shape, new_strides};
        return std::shared_ptr<Tensor>(new Tensor(meta, this->_storage, this->_offset));
    }
    std::vector<ptrdiff_t> new_strides;new_strides.reserve(shape.size());
    const auto&old_strides=this->strides();
    const auto&old_shape=this->shape();
    size_t old_dim_index=0,split_divisor=1;
    for(size_t new_dim_index=0;new_dim_index<shape.size();new_dim_index++){
        if(shape[new_dim_index]==1){
            new_strides.emplace_back(1);
            continue;
        }
        size_t target_dim=shape[new_dim_index];
        if(old_dim_index>=old_shape.size()){
            throw std::runtime_error("Dim Error!");
        }
        size_t available_dim_size=old_shape[old_dim_index]/split_divisor;
        size_t original_stride=old_strides[old_dim_index];
        if(target_dim<available_dim_size){
            if(available_dim_size%target_dim!=0) throw std::runtime_error("Split Error!");
            ptrdiff_t new_strides_val=(available_dim_size/target_dim)*original_stride;
            new_strides.emplace_back(new_strides_val);
            split_divisor*=target_dim;
            continue;
        }
        size_t accumulated_size=available_dim_size;
        split_divisor=1;
        old_dim_index++;
        while(accumulated_size<target_dim){
            if(old_dim_index>=old_shape.size()){
                throw std::runtime_error("Dim Error!");
            }
            if(accumulated_size>1){
                if(old_strides[old_dim_index-1]!=static_cast<ptrdiff_t>(old_shape[old_dim_index]*old_strides[old_dim_index])){
                    throw std::runtime_error("Transform Error!");
                }
            }  
            accumulated_size*=old_shape[old_dim_index++];
        }
        if(accumulated_size!=target_dim){ 
            throw std::runtime_error("Match Error!");
        }
        new_strides.emplace_back(old_strides[old_dim_index-1]);
    }
    TensorMeta _meta{this->dtype(),shape,std::move(new_strides)};
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage,this->_offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    if(dim>=this->ndim()) throw std::runtime_error("Dim Error!");
    if(start>=end) throw std::runtime_error("Index Error!");
    if(end>this->shape()[dim]) throw std::runtime_error("End Error!");
    auto new_shape=this->shape();
    new_shape[dim]=end-start;
    auto new_offset=this->_offset+start*this->strides()[dim]*this->elementSize();
    TensorMeta _meta{this->dtype(),std::move(new_shape),this->strides()};
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage,new_offset));
}

void Tensor::load(const void *src_) {
    std::byte* dis_ptr=this->data();
    size_t Elemsize_in_bytes=this->elementSize()*this->numel();
    llaisysMemcpyKind_t CurKind=this->deviceType()==LLAISYS_DEVICE_CPU?LLAISYS_MEMCPY_H2H:LLAISYS_MEMCPY_H2D;
    core::context().runtime().api()->memcpy_sync(dis_ptr,src_,Elemsize_in_bytes,CurKind);
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
