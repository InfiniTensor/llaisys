#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>
#include <stdexcept>


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
    const auto &shp = this->shape();
    const auto &str = this->strides();
    const size_t n = shp.size();

    // Empty or scalar tensor is contiguous
    if (n == 0 || this->numel() == 0) {
        return true;
    }

    size_t expected = 1;
    // Walk from the last dimension to the first
    for (size_t i = n; i-- > 0;) {
        // Dimensions with size 0 or 1 do not affect contiguity
        if (shp[i] <= 1) {
            continue;
        }
        // Strides are in units of elements
        if ((size_t)str[i] != expected) {
            return false;
        }
        expected *= shp[i];
    }
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    const auto &old_shape = this->shape();
    const auto &old_strides = this->strides();
    const size_t ndim = old_shape.size();

    if (order.size() != ndim) {
        throw std::invalid_argument("Tensor::permute: order size mismatch");
    }

    // Validate order is a permutation of [0, ndim)
    std::vector<uint8_t> used(ndim, 0);
    for (size_t idx : order) {
        if (idx >= ndim) {
            throw std::invalid_argument("Tensor::permute: index out of range");
        }
        if (used[idx]) {
            throw std::invalid_argument("Tensor::permute: duplicate index in order");
        }
        used[idx] = 1;
    }

    // Build new shape and strides (strides are in elements)
    std::vector<size_t> new_shape(ndim);
    std::vector<ptrdiff_t> new_strides(ndim);
    for (size_t i = 0; i < ndim; ++i) {
        const size_t src = order[i];
        new_shape[i] = old_shape[src];
        new_strides[i] = old_strides[src];
    }

    TensorMeta meta{this->dtype(), new_shape, new_strides};
    return std::shared_ptr<Tensor>(new Tensor(meta, _storage, _offset));
}

tensor_t Tensor::view(const std::vector<size_t> &new_shape) const {
    const auto &old_shape   = this->shape();
    const auto &old_strides = this->strides();
    const size_t old_ndim   = old_shape.size();

    // 1) Number of elements must match (allow numel==0)
    const size_t old_numel = this->numel();
    size_t new_numel = 1;
    for (auto s : new_shape) {
        new_numel *= s;
    }
    if ((old_numel != 0 || new_numel != 0) && (old_numel != new_numel)) {
        throw std::invalid_argument("Tensor::view: number of elements mismatch");
    }

    // Helper to build standard contiguous strides for any shape
    auto make_contig_strides = [](const std::vector<size_t> &shp) {
        std::vector<ptrdiff_t> st(shp.size(), 0);
        size_t running = 1;
        for (size_t i = shp.size(); i-- > 0; ) {
            st[i] = static_cast<ptrdiff_t>(running);
            size_t dim = shp[i] == 0 ? 1 : shp[i];
            running *= dim;
        }
        return st;
    };

    // For numel <= 1, any view is fine and can use standard contiguous strides
    if (old_numel <= 1) {
        TensorMeta meta{this->dtype(), new_shape, make_contig_strides(new_shape)};
        return std::shared_ptr<Tensor>(new Tensor(meta, _storage, _offset));
    }

    // 2) Collapse old dims (right to left) into contiguous blocks
    struct Block { size_t size; size_t stride; }; // stride is in elements
    std::vector<Block> blocks;
    for (size_t rev = 0; rev < old_ndim; ++rev) {
        size_t i = old_ndim - 1 - rev;
        const size_t dim = old_shape[i];
        if (dim <= 1) continue;
        const size_t st = static_cast<size_t>(old_strides[i]);

        if (blocks.empty()) {
            blocks.push_back({dim, st});
        } else {
            Block &last = blocks.back();
            // Contiguous if stride[i] == last.stride * last.size
            if (st == last.stride * last.size) {
                last.size *= dim; // merge into the same block
            } else {
                blocks.push_back({dim, st}); // start a new block
            }
        }
    }

    // 3) Fit new_shape (right to left) into those blocks and build new strides
    std::vector<ptrdiff_t> new_strides(new_shape.size(), 0);

    if (blocks.empty()) {
        // Should not happen for numel > 1, but fall back to contiguous strides
        TensorMeta meta{this->dtype(), new_shape, make_contig_strides(new_shape)};
        return std::shared_ptr<Tensor>(new Tensor(meta, _storage, _offset));
    }

    size_t bidx = blocks.size() - 1;   // current old block (from the right)
    size_t need = blocks[bidx].size;   // how many elements remain to fill in this block
    size_t base = blocks[bidx].stride; // base stride (in elements) for this block

    bool first_in_block = true;
    for (size_t k = new_shape.size(); k-- > 0; ) {
        const size_t s = new_shape[k];

        // Assign stride for this new dim within the current block
        if (first_in_block) {
            new_strides[k] = static_cast<ptrdiff_t>(base);
            first_in_block = false;
        } else {
            size_t next_len = new_shape[k + 1];
            if (next_len == 0) next_len = 1;
            new_strides[k] = static_cast<ptrdiff_t>(static_cast<size_t>(new_strides[k + 1]) * next_len);
        }

        // Consume this dim within the current block (s==1 does not change "need")
        if (s > 1) {
            if (need % s != 0) {
                throw std::invalid_argument("Tensor::view: shape not compatible with memory layout");
            }
            need /= s;
        }

        // If the current block is exactly filled, move to the next block (to the left)
        if (need == 1) {
            if (bidx == 0) {
                // No more old blocks. Remaining new dims (if any) must be size==1.
                first_in_block = true;
            } else {
                --bidx;
                need = blocks[bidx].size;
                base = blocks[bidx].stride;
                first_in_block = true;
            }
        }
    }

    // All old blocks must be exactly matched
    if (!(bidx == 0 && need == 1)) {
        throw std::invalid_argument("Tensor::view: new shape does not align to contiguous blocks");
    }

    TensorMeta meta{this->dtype(), new_shape, new_strides};
    return std::shared_ptr<Tensor>(new Tensor(meta, _storage, _offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    const auto &old_shape = this->shape();
    const auto &old_strides = this->strides();
    const size_t ndim = old_shape.size();

    // 1) Validate args
    if (dim >= ndim) {
        throw std::invalid_argument("Tensor::slice: dim out of range");
    }
    if (start > end) {
        throw std::invalid_argument("Tensor::slice: start must be <= end");
    }
    if (end > old_shape[dim]) {
        throw std::invalid_argument("Tensor::slice: end exceeds dimension size");
    }
    if (old_strides[dim] < 0) {
        throw std::invalid_argument("Tensor::slice: negative stride not supported");
    }

    // 2) Compute new offset (strides are in elements)
    const size_t esize = this->elementSize();
    const size_t step_elems = static_cast<size_t>(old_strides[dim]);
    const size_t new_offset = _offset + start * step_elems * esize;

    // 3) New shape/strides
    std::vector<size_t> new_shape = old_shape;
    std::vector<ptrdiff_t> new_strides = old_strides;
    new_shape[dim] = end - start;

    // 4) Build and return
    TensorMeta meta{this->dtype(), new_shape, new_strides};
    return std::shared_ptr<Tensor>(new Tensor(meta, _storage, new_offset));
}

void Tensor::load(const void *src_) {
    if (src_ == nullptr) {
        throw std::invalid_argument("Tensor::load: src is null");
    }
    if (!_storage) {
        throw std::runtime_error("Tensor::load: storage is null");
    }

    const size_t nbytes = this->numel() * this->elementSize();

    if (_offset + nbytes > _storage->size()) {
        throw std::out_of_range("Tensor::load: out of storage bounds");
    }

    // Ensure runtime uses the correct device
    core::context().setDevice(this->deviceType(), this->deviceId());
    auto *api = core::context().runtime().api();

    // If you want to enforce contiguous only, uncomment this:
    // if (!this->isContiguous()) {
    //     throw std::runtime_error("Tensor::load: only contiguous tensor supported for now");
    // }

    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        // Host -> Host
        std::memcpy(this->data(), src_, nbytes);
    } else {
        // Host -> Device
        api->memcpy_sync(this->data(), src_, nbytes, LLAISYS_MEMCPY_H2D);
        // If memcpy is async in your backend, you may synchronize here.
        // api->device_synchronize();
    }
}

tensor_t Tensor::contiguous() const {
    //TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    //TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    //TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

} // namespace llaisys
