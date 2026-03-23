#pragma once
#include "../core/llaisys_core.hpp"

#include <vector>
namespace llaisys {
    //前向声明张量类
    class Tensor;
    //张量的共享指针类型
    using tensor_t = std::shared_ptr<Tensor>;

    //描述张量形状、数据类型和步长的元数据
    struct TensorMeta {
        //数据类型
        llaisysDataType_t dtype;
        //形状
        std::vector<size_t> shape;
        //步长
        std::vector<ptrdiff_t> strides;
    };

    //张量
    class Tensor {
    private:
        //描述张量形状、数据类型和步长的元数据
        TensorMeta _meta;
        //指向存储张量数据的内存块的共享指针。它可以被多个张量共享。有关更多详细信息，请查看storage类   
        core::storage_t _storage;
        //张量在存储中的起始索引（以字节为单位）
        size_t _offset;

        //构造器
        Tensor(TensorMeta meta, core::storage_t storage, size_t offset = 0);

    public:
        //创建一个新的张量
        static tensor_t create(
            //张量形状
            const std::vector<size_t> &shape,
            //数据类型
            llaisysDataType_t dtype,
            //默认在CPU上创建张量
            llaisysDeviceType_t device_type = LLAISYS_DEVICE_CPU,
            //设备ID，默认为0
            int device = 0);
        //析构器
        ~Tensor() = default;
        // Info
        //返回指向张量数据的指针
        std::byte *data();
        //返回指向张量数据的常量指针
        const std::byte *data() const;
        //返回张量的维度数
        size_t ndim() const;
        //返回张量的形状
        const std::vector<size_t> &shape() const;
        //返回张量的步长    
        const std::vector<ptrdiff_t> &strides() const;
        //返回张量的数据类型
        llaisysDataType_t dtype() const;
        //返回张量所存储数据的存储对象
        llaisysDeviceType_t deviceType() const;
        //返回张量所在设备的ID
        int deviceId() const;
        //返回张量中元素的总数
        size_t numel() const;
        //返回张量中每个元素的大小（以字节为单位）
        size_t elementSize() const;

        //调试信息
        std::string info() const;
        //打印张量的调试信息
        void debug() const;
        //检查张量是否是连续存储的
        bool isContiguous() const;

        // Meta Transform
        tensor_t permute(const std::vector<size_t> &order) const;
        tensor_t slice(size_t dim, size_t start, size_t end) const;
        tensor_t view(const std::vector<size_t> &shape) const;

        // Load data from host memory
        void load(const void *src);

        // Challenging features
        tensor_t contiguous() const;
        tensor_t reshape(const std::vector<size_t> &shape) const;
        tensor_t to(llaisysDeviceType_t device_type, int device = -1) const;
    };

} // namespace llaisys
