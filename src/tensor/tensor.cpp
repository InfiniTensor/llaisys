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

    // 检查 tensor 是否连续存储
    // 连续存储意味着相邻元素在内存中也是相邻的
    const auto &shape = _meta.shape;
    const auto &strides = _meta.strides;

    // 空 tensor 视为连续
    if (shape.empty()) {
        return true;
    }

    // 从最后一维开始，期望的 stride 初始为 1
    ptrdiff_t expected = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        // 跳过 size=1 的维度，因为它们不影响连续性
        if (shape[i] != 1) {
            // 检查当前维度的 stride 是否等于期望值
            if (strides[i] != expected) {
                return false;
            }
            // 更新下一维的期望 stride
            expected *= static_cast<ptrdiff_t>(shape[i]);
        }
    }
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    // return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
    const size_t ndims = _meta.shape.size();

    CHECK_ARGUMENT(order.size() == ndims, "permute: order size mismatch");

    // valid order
    std::vector<bool> seen(ndims, false);
    for (size_t idx:order) {
        CHECK_ARGUMENT(idx < ndims, "permute: index out of range");
        CHECK_ARGUMENT(!seen[idx], "permute: duplicate index in order");
        seen[idx] = true;
    }

    // build new meta
    TensorMeta new_meta;
    new_meta.dtype = _meta.dtype;
    new_meta.shape.resize(ndims);
    new_meta.strides.resize(ndims);

    for (size_t i = 0;i < ndims; ++i) {
        new_meta.shape[i] = _meta.shape[order[i]];
        new_meta.strides[i] = _meta.strides[order[i]];
    }

    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));

}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    // return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
    // calculate  total of new elements
    size_t new_numel = 1;
    for (size_t s:shape) {
        new_numel *= s;
    }

    CHECK_ARGUMENT(new_numel == numel(), "view: element count mismatch");
    CHECK_ARGUMENT(isContiguous(), "view: tensor must be contiguous");

    // build  new meta
    TensorMeta new_meta;
    new_meta.dtype = _meta.dtype;
    new_meta.shape = shape;

    // calculate strides (Tensor::create)
    size_t ndim_ = shape.size();
    new_meta.strides.resize(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        new_meta.strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }

    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    // return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
    // valid dim
    CHECK_ARGUMENT(dim < _meta.shape.size(), "slice: dim out of range");
    CHECK_ARGUMENT(start < end, "slice: start must be less than end");
    CHECK_ARGUMENT(end <= _meta.shape[dim], "slice: end exceeds dimension size");

    // copy meta and revise shape
    TensorMeta new_meta = _meta;
    new_meta.shape[dim] = end - start;

    // calculate the new offset
    // start * stride[dim] * each_elements_bytes
    size_t new_offset = _offset + 
            start * static_cast<size_t>(_meta.strides[dim])* elementSize();

    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, new_offset));
}

void Tensor::load(const void *src_) {
    //std::cerr << "[DBG] enter Tensor::load, src=" << src_ << std::endl;
    CHECK_ARGUMENT(src_ != nullptr, "load: src is null");
    //std::cerr << "[DBG] after CHECK_ARGUMENT" << std::endl;

    void *dst = data();
    size_t size = numel() * elementSize();
    // std::cout << "memcpy_sync: " << size << " bytes" << std::endl;
    // std::cout << "memcpy_sync: " << src_ << " -> " << dst << std::endl;
    // std::cout << "memcpy_sync: " << deviceType() << std::endl;
    device::getRuntimeAPI(deviceType())->memcpy_sync(dst, src_, size, LLAISYS_MEMCPY_H2D);
    auto api = device::getRuntimeAPI(deviceType());
    //std::cerr << "[DBG] api=" << api << " deviceType=" << deviceType() << std::endl;
    api->memcpy_sync(dst, src_, size, LLAISYS_MEMCPY_H2D);
    core::context().runtime().api()->device_synchronize();
}

tensor_t Tensor::contiguous() const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
    
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
    
    // return contiguous()->view(shape);
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

} // namespace llaisys