#include "tensor.hpp"
#include "../utils.hpp"
#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

// 私有构造函数：初始化 _meta, _storage-存储, _offset-偏移量, 
// 私有构造函数：只能通过工厂方法创建，确保对象创建可控
// 初始化列表：在构造函数体执行前初始化成员变量
// std::move：转移所有权，避免不必要的拷贝
Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)),
    _storage(std::move(storage)),
    _offset(offset) {}

// create() 工厂方法:
// 计算步长（strides）- 用于行主序（row-major）内存布局
// 判断设备类型：若请求CPU但当前运行时是GPU，分配主机内存；否则根据设备类型分配存储
tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    // 计算行主序步长 - 跳步函数 strides-记录每一个维度要跳的大小
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);


    // 2. 内存分配
    // 如果请求 CPU 但当前环境不是 CPU，强制分配 Host 内存
    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        // 否则在指定设备上分配内存
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}
// 3.2 数据访问 (data 函数)
// 这里体现了 _offset 的作用。如果这个张量是另一个大张量的一部分（切片），
// _storage->memory() 指向大张量的开头，而 + _offset 让指针正确指向切片的开始位置
std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

// ndim()    维度数
size_t Tensor::ndim() const {
    return _meta.shape.size();
}
// shape()   形状向量
const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}
// strides()   步长向量
const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}
// dtype()     数据类型
llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}
// deviceType()     设备类型
llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}
// deviceId()     设备ID
int Tensor::deviceId() const {
    return _storage->deviceId();
}
// numel()：计算总元素数（形状中所有维度相乘
size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}
// elementSize()：单个元素的字节大小
size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}
// info()：返回张量信息的字符串表示
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

// print_data<T>()：递归打印多维数据，支持不同数据类型
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

// debug_print()：根据数据类型转发到 print_data()
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

// 3.3 调试与打印 (debug 和 print_data)
// 递归打印 (print_data): 这是一个模板函数，
// 通过递归方式处理任意维度的张量打印。
// 当 dim 到达最后一维时打印数值，否则递归调用下一维
void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    // ... 打印元信息 ...
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        // 如果是 CPU 张量，直接读取内存打印
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        // 如果是 GPU 张量，不能直接读取！
        // 1. 创建一个临时的 CPU 张量
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        // 2. 将数据从设备拷贝到主机 (D2H)
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        // 3. 打印临时张量的数据
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}
// isContiguous()：检查内存连续性
bool Tensor::isContiguous() const {
    TO_BE_IMPLEMENTED();
    return true;
}
// permute()：维度重排
tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}
// view()：改变形状（不复制数据）
tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}
// slice()：切片操作
tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

/// *********************************************** ///
// load() 函数：将主机内存数据加载到张量
// 计算字节数,切换到张量所在设备
// CPU设备：直接 memcpy,其他设备：异步 H2D（主机→设备）复制，然后同步等待
void Tensor::load(const void *src_) {
    // 计算需要复制的字节数
    size_t bytes = numel() * elementSize();

    // 如果 src_ 为 nullptr，直接返回
    if (src_ == nullptr || bytes == 0) {
        return;
    }

    // 切换到张量所在设备上下文（确保 runtime/api 是对应设备的）
    core::context().setDevice(this->deviceType(), this->deviceId());

    // 如果底层存储是主机内存（或张量在 CPU 上），直接 memcpy
    if (_storage->isHost() || this->deviceType() == LLAISYS_DEVICE_CPU) {
        std::memcpy(this->data(), src_, bytes);
        return;
    }

    // 否则使用运行时提供的同步主机->设备拷贝（H2D）
    const auto *api = core::context().runtime().api();
    api->memcpy_sync(this->data(), src_, bytes, LLAISYS_MEMCPY_H2D);
}

// contiguous()：转为连续内存布局
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
