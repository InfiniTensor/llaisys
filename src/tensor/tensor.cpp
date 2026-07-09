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
    //first initial expected stride(from lastdim to 0)
    //test
    ptrdiff_t expected_stride = 1;
    for(int i = (int)ndim()-1; i >= 0; i--)
    {
        size_t current_shape = shape()[i];
        if(current_shape == 0) return true;
        if(current_shape > 1)
        {
            if(strides()[i] != expected_stride)
                return false;
            expected_stride *= current_shape;
        }
    }
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    // valid order.size() == this->ndim()
    CHECK_ARGUMENT(order.size() == this->ndim(), "Permute: order size mismatch.");
    std::vector<bool> seen(order.size(), false);
    std::vector<size_t> new_shape(order.size());
    std::vector<ptrdiff_t> new_strides(order.size());
    for(int i = 0; i < (int)order.size(); ++i)
    {
        size_t dim = order[i];
        CHECK_ARGUMENT(dim < this->ndim(), "Permute: order index out of range.");
        CHECK_ARGUMENT(!seen[dim], "Permute: order index duplicated.");
        seen[dim] = true;
        new_shape[i] = this->shape()[dim];
        new_strides[i] = this->strides()[dim];
    } 
    TensorMeta meta{this->dtype(), new_shape, new_strides};
    return std::shared_ptr<Tensor>(new Tensor(meta, _storage, _offset));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {

    // calculate new_shape's numel, must be the same as the old
    size_t new_numel = 1;
    for(auto s : shape)
        new_numel *= s;
    CHECK_ARGUMENT(new_numel == this->numel(), "View: total elements mismatch.");
    std::vector<ptrdiff_t> new_strides(shape.size(),0);

    //filter size = 1 shape in old_shape
    std::vector<size_t> old_shape;
    std::vector<ptrdiff_t> old_strides;
    old_shape.reserve(this->shape().size());
    old_strides.reserve(this->strides().size());
    for(size_t i = 0; i < this->shape().size(); ++i)
    {
        if(this->shape()[i] > 1)
        {
            old_shape.push_back(this->shape()[i]);
            old_strides.push_back(this->strides()[i]);
        }
    }

    // filter size = 1 shape in new_shape
    std::vector<size_t> new_shape;
    std::vector<size_t> new_indices;
    new_shape.reserve(shape.size());
    new_indices.reserve(shape.size());
    for(size_t i = 0; i < shape.size(); ++i)
    {
        if(shape[i] > 1)
        {
            new_shape.push_back(shape[i]);
            new_indices.push_back(i);
        }
    }

    // if no size > 1 in old_shape or new_shape
    // no need to spilt blocks, generate continual new_strides
    if(old_shape.empty() || new_shape.empty())
    {
        for(int i = (int)shape.size()-1; i>=0; --i)
        {
            if(i+1 < (int)shape.size())
            {
                new_strides[i] = new_strides[i+1]*shape[i+1];
            }
            else
            {
                new_strides[i] = 1;
            }
        }
        TensorMeta meta{this->dtype(), shape, new_strides};
        return std::shared_ptr<Tensor>(new Tensor(meta, _storage));
    }
    // spilt blocks
    struct Block
    {
        size_t numel;
        ptrdiff_t inner_stride;
    };

    std::vector<Block> blocks;
    for(int i = (int)old_shape.size() - 1; i >= 0; --i)
    {
        if(blocks.empty())
        {
            blocks.push_back({old_shape[i], old_strides[i]});
        }
        else
        {
            //continual add to current block
            if(old_strides[i] == (ptrdiff_t)old_shape[i+1] * old_strides[i+1])
            {
                blocks.back().numel *= old_shape[i];
            }
            //break, spilt new block
            else
            {
                blocks.push_back({old_shape[i], old_strides[i]});
            }
        }
    }
    // for every block, judge which dims can cover it
    size_t dim_pos = 0;
    for(int b = (int)blocks.size() - 1; b >= 0; --b)
    {
        const auto &block = blocks[b];
        size_t start = dim_pos;
        size_t prod = 1;
        while(dim_pos < new_shape.size() && prod < block.numel)
        {
            prod *= new_shape[dim_pos];
            dim_pos++;
        }
        CHECK_ARGUMENT(prod == block.numel, "View: shape is not compatible with storage.");
        // set new_strides for this block
        ptrdiff_t stride = block.inner_stride;
        for(int j = (int)dim_pos - 1; j >= (int)start; --j)
        {
            new_strides[new_indices[j]] = stride;
            stride *= (ptrdiff_t)new_shape[j];
        }
    }
    CHECK_ARGUMENT(dim_pos == new_shape.size(), "View: shape is not compatible with storage.");
    // for size = 1 dims, set new_strides[i] = new_strides[i+1] * shape[i+1]
    for(int i = (int)shape.size() - 1; i >= 0; --i)
    {
        if(shape[i] == 1)
        {
            if(i + 1 < (int)shape.size())
            {
                new_strides[i] = new_strides[i+1] * shape[i+1];
            }
            else
            {
                new_strides[i] = 1;
            }
        }
    }
    TensorMeta meta{this->dtype(), shape, new_strides};
    return std::shared_ptr<Tensor>(new Tensor(meta, _storage, _offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    CHECK_ARGUMENT(dim < this->ndim(), "Slice: dim out of range.");
    CHECK_ARGUMENT(start <= end, "Slice: start must be less than or equal to end.");
    CHECK_ARGUMENT(end <= this->shape()[dim], "Slice: end out of range.");
    std::vector<size_t> new_shape = this->shape();
    new_shape[dim] = end - start;
    std::vector<ptrdiff_t> new_strides = this->strides();
    size_t byte_offset = _offset + start * (size_t)new_strides[dim] * this->elementSize();
    TensorMeta meta{this->dtype(), new_shape, new_strides};
    return std::shared_ptr<Tensor>(new Tensor(meta, _storage, byte_offset));
}

void Tensor::load(const void *src_) {
    CHECK_ARGUMENT(src_ != nullptr, "Load: src is null.");
    size_t size = this->numel() * this->elementSize();
    core::context().setDevice(this->deviceType(), this->deviceId());
    if (!_storage || _offset + size > _storage->size()) {
        if (this->deviceType() == LLAISYS_DEVICE_CPU) {
            _storage = core::context().runtime().allocateHostStorage(size);
        } else {
            _storage = core::context().runtime().allocateDeviceStorage(size);
        }
        _offset = 0;
    }
    ASSERT(this->isContiguous(), "Load: tensor must be contiguous.");
    core::context().runtime().api()->memcpy_sync(
        this->data(), src_, size,
        this->deviceType() == LLAISYS_DEVICE_CPU ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_H2D);
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
