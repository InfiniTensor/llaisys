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
    ptrdiff_t expected = 1;
    for (size_t i = _meta.shape.size(); i > 0; --i) {
        if (_meta.strides[i - 1] != expected) {
            return false;
        }
        expected *= static_cast<ptrdiff_t>(_meta.shape[i - 1]);
    }
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    ASSERT(order.size() == ndim(), "Permute: order must have same number of dimensions.");
    TensorMeta new_meta;
    new_meta.dtype = _meta.dtype;
    new_meta.shape.resize(order.size());
    new_meta.strides.resize(order.size());
    for (size_t i = 0; i < order.size(); ++i) {
        ASSERT(order[i] < ndim(), "Permute: order index out of range.");
        new_meta.shape[i] = _meta.shape[order[i]];
        new_meta.strides[i] = _meta.strides[order[i]];
    }
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    size_t new_numel = 1;
    for (auto s : shape) new_numel *= s;
    ASSERT(new_numel == numel(), "View: new shape must have the same number of elements.");

    size_t new_ndim = shape.size();
    std::vector<ptrdiff_t> new_strides(new_ndim);

    if (new_numel == 0) {
        ptrdiff_t s = 1;
        for (size_t i = new_ndim; i > 0; --i) {
            new_strides[i - 1] = s;
            s *= static_cast<ptrdiff_t>(shape[i - 1]);
        }
        TensorMeta new_meta{_meta.dtype, shape, new_strides};
        return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
    }

    // Filter out size-1 dims from old shape
    std::vector<size_t> old_sh;
    std::vector<ptrdiff_t> old_st;
    for (size_t i = 0; i < ndim(); i++) {
        if (_meta.shape[i] != 1) {
            old_sh.push_back(_meta.shape[i]);
            old_st.push_back(_meta.strides[i]);
        }
    }
    // Filter out size-1 dims from new shape, remember original indices
    std::vector<size_t> new_sh;
    std::vector<size_t> new_map;
    for (size_t i = 0; i < new_ndim; i++) {
        if (shape[i] != 1) {
            new_sh.push_back(shape[i]);
            new_map.push_back(i);
        }
    }

    size_t oi = 0, ni = 0;
    while (oi < old_sh.size() && ni < new_sh.size()) {
        size_t op = old_sh[oi], np = new_sh[ni];
        size_t ni_start = ni;

        while (op != np) {
            if (op < np) {
                ++oi;
                ASSERT(oi < old_sh.size(), "View: incompatible shapes.");
                ASSERT(old_st[oi - 1] == old_st[oi] * static_cast<ptrdiff_t>(old_sh[oi]),
                       "View: cannot view a non-contiguous tensor.");
                op *= old_sh[oi];
            } else {
                ++ni;
                ASSERT(ni < new_sh.size(), "View: incompatible shapes.");
                np *= new_sh[ni];
            }
        }

        // Fill strides for new dims [ni_start..ni] right-to-left
        ptrdiff_t s = old_st[oi];
        for (size_t k = ni + 1; k > ni_start; --k) {
            new_strides[new_map[k - 1]] = s;
            s *= static_cast<ptrdiff_t>(new_sh[k - 1]);
        }
        ++oi;
        ++ni;
    }

    // Fill strides for size-1 dims in the new shape
    for (int i = static_cast<int>(new_ndim) - 1; i >= 0; --i) {
        if (shape[i] == 1) {
            new_strides[i] = (i + 1 < static_cast<int>(new_ndim))
                                 ? new_strides[i + 1] * static_cast<ptrdiff_t>(shape[i + 1])
                                 : 1;
        }
    }

    TensorMeta new_meta{_meta.dtype, shape, new_strides};
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    ASSERT(dim < ndim(), "Slice: dim out of range.");
    ASSERT(start < end && end <= _meta.shape[dim], "Slice: invalid range.");

    TensorMeta new_meta = _meta;
    new_meta.shape[dim] = end - start;

    size_t new_offset = _offset + start * static_cast<size_t>(_meta.strides[dim]) * elementSize();
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, new_offset));
}

void Tensor::load(const void *src_) {
    size_t bytes = numel() * elementSize();
    if (deviceType() == LLAISYS_DEVICE_CPU) {
        core::context().setDevice(LLAISYS_DEVICE_CPU, 0);
        core::context().runtime().api()->memcpy_sync(data(), src_, bytes, LLAISYS_MEMCPY_H2H);
    } else {
        core::context().setDevice(deviceType(), deviceId());
        core::context().runtime().api()->memcpy_sync(data(), src_, bytes, LLAISYS_MEMCPY_H2D);
    }
}

tensor_t Tensor::contiguous() const {
    if (isContiguous()) {
        return std::shared_ptr<Tensor>(new Tensor(_meta, _storage, _offset));
    }
    auto result = create(shape(), dtype(), deviceType(), deviceId());
    // Use rearrange: copy data from non-contiguous to contiguous
    // We need to do element-wise copy respecting strides
    core::context().setDevice(deviceType(), deviceId());
    size_t n = numel();
    size_t esize = elementSize();
    size_t nd = ndim();
    auto &sh = _meta.shape;
    auto &st = _meta.strides;

    if (deviceType() == LLAISYS_DEVICE_CPU) {
        std::vector<size_t> idx(nd, 0);
        for (size_t i = 0; i < n; ++i) {
            ptrdiff_t src_off = 0;
            for (size_t d = 0; d < nd; ++d) src_off += idx[d] * st[d];
            std::memcpy(result->data() + i * esize, data() + src_off * esize, esize);
            for (int d = static_cast<int>(nd) - 1; d >= 0; --d) {
                if (++idx[d] < sh[d]) break;
                idx[d] = 0;
            }
        }
    } else {
        auto api = core::context().runtime().api();
        // For GPU: use element-wise copy with strides via device memcpy
        // Copy to CPU, make contiguous there, copy back
        auto cpu_src = to(LLAISYS_DEVICE_CPU, 0);
        auto cpu_contig = cpu_src->contiguous();
        api->memcpy_sync(result->data(), cpu_contig->data(), n * esize, LLAISYS_MEMCPY_H2D);
    }
    return result;
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    if (isContiguous()) {
        return view(shape);
    }
    return contiguous()->view(shape);
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    if (device_type == deviceType() && device == deviceId()) {
        return std::shared_ptr<Tensor>(new Tensor(_meta, _storage, _offset));
    }

    auto src = isContiguous() ? std::shared_ptr<Tensor>(new Tensor(_meta, _storage, _offset)) : contiguous();
    auto dst = create(shape(), dtype(), device_type, device);
    size_t bytes = numel() * elementSize();

    llaisysMemcpyKind_t kind;
    if (deviceType() == LLAISYS_DEVICE_CPU && device_type != LLAISYS_DEVICE_CPU) {
        kind = LLAISYS_MEMCPY_H2D;
        core::context().setDevice(device_type, device);
    } else if (deviceType() != LLAISYS_DEVICE_CPU && device_type == LLAISYS_DEVICE_CPU) {
        kind = LLAISYS_MEMCPY_D2H;
        core::context().setDevice(deviceType(), deviceId());
    } else if (deviceType() != LLAISYS_DEVICE_CPU && device_type != LLAISYS_DEVICE_CPU) {
        kind = LLAISYS_MEMCPY_D2D;
        core::context().setDevice(deviceType(), deviceId());
    } else {
        kind = LLAISYS_MEMCPY_H2H;
        core::context().setDevice(LLAISYS_DEVICE_CPU, 0);
    }

    core::context().runtime().api()->memcpy_sync(dst->data(), src->data(), bytes, kind);
    return dst;
}

} // namespace llaisys
