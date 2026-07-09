#pragma once
#include "../core/llaisys_core.hpp"

#include <vector>
namespace llaisys {
class Tensor;
using tensor_t = std::shared_ptr<Tensor>;

struct TensorMeta {
    llaisysDataType_t dtype;
    std::vector<size_t> shape;
    std::vector<ptrdiff_t> strides;
};

class Tensor {
private:
    TensorMeta _meta;
    core::storage_t _storage;
    size_t _offset;
    Tensor(TensorMeta meta, core::storage_t storage, size_t offset = 0);

public:
    static tensor_t create(
        const std::vector<size_t> &shape,
        llaisysDataType_t dtype,
        llaisysDeviceType_t device_type = LLAISYS_DEVICE_CPU,
        int device = 0);
    ~Tensor() = default;
    // Info
    std::byte *data();             // 指向张量数据的指针
    const std::byte *data() const; // 指向张量数据的指针
    size_t ndim() const;           // 维度数量
    const std::vector<size_t> &shape() const;
    const std::vector<ptrdiff_t> &strides() const;
    llaisysDataType_t dtype() const;        // 数据类型
    llaisysDeviceType_t deviceType() const; // 设备类型
    int deviceId() const;                   // 设备ID
    size_t numel() const;                   // 元素总数
    size_t elementSize() const;             // 每个元素的大小（字节）

    std::string info() const;
    void debug() const;

    // shape 和 strides 决定了张量在内存中的布局
    // 判断张量是否是连续存储的
    // 最后一维的步长为一，倒数第二维的步长为最后一维的大小，依此类推
    bool isContiguous() const;

    // Meta Transform
    tensor_t permute(const std::vector<size_t> &order) const;
    tensor_t slice(size_t dim, size_t start, size_t end) const;
    tensor_t view(const std::vector<size_t> &shape) const;

    // Load data from host memory
    void load(const void *src);
    // src_: 指向源数据的指针, 实际是主机内存中的数据
    // 将数据加载到张量中，如果张量在设备上，则进行相应的内存拷贝；否则直接拷贝

    // Challenging features
    tensor_t contiguous() const;
    tensor_t reshape(const std::vector<size_t> &shape) const;
    tensor_t to(llaisysDeviceType_t device_type, int device = -1) const;
};

} // namespace llaisys
