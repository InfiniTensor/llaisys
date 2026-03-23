#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../device/runtime_api.hpp"

#include "cpu/rearrange_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/rearrange_nvidia.hpp"
#endif
#ifdef ENABLE_ILUVATAR_API
#include "nvidia/rearrange_nvidia.hpp"
#endif

namespace llaisys::ops {
void rearrange(tensor_t out, tensor_t in) {
    CHECK_SAME_DEVICE(out, in);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    ASSERT(out->shape() == in->shape(), "Rearrange: shapes must match.");

    const auto elem_size = out->elementSize();
    const auto &shape = out->shape();
    const auto &out_strides = out->strides();
    const auto &in_strides = in->strides();

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rearrange(out->data(), in->data(), shape, out_strides, in_strides, elem_size);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rearrange(out->data(), in->data(), shape, out_strides, in_strides, elem_size);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
    {
        const auto runtime = llaisys::device::getRuntimeAPI(out->deviceType());
        const size_t ndim = shape.size();
        const size_t shape_bytes = ndim * sizeof(size_t);
        const size_t stride_bytes = ndim * sizeof(ptrdiff_t);
        void *shape_dev = runtime->malloc_device(shape_bytes);
        void *out_strides_dev = runtime->malloc_device(stride_bytes);
        void *in_strides_dev = runtime->malloc_device(stride_bytes);
        runtime->memcpy_sync(shape_dev, shape.data(), shape_bytes, LLAISYS_MEMCPY_H2D);
        runtime->memcpy_sync(out_strides_dev, out_strides.data(), stride_bytes, LLAISYS_MEMCPY_H2D);
        runtime->memcpy_sync(in_strides_dev, in_strides.data(), stride_bytes, LLAISYS_MEMCPY_H2D);
        nvidia::rearrange(out->data(), in->data(),
                          reinterpret_cast<const size_t *>(shape_dev),
                          reinterpret_cast<const ptrdiff_t *>(out_strides_dev),
                          reinterpret_cast<const ptrdiff_t *>(in_strides_dev),
                          ndim, elem_size, out->numel());
        runtime->free_device(shape_dev);
        runtime->free_device(out_strides_dev);
        runtime->free_device(in_strides_dev);
        return;
    }
#endif
#ifdef ENABLE_ILUVATAR_API
    case LLAISYS_DEVICE_ILUVATAR:
    {
        const auto runtime = llaisys::device::getRuntimeAPI(out->deviceType());
        const size_t ndim = shape.size();
        const size_t shape_bytes = ndim * sizeof(size_t);
        const size_t stride_bytes = ndim * sizeof(ptrdiff_t);
        void *shape_dev = runtime->malloc_device(shape_bytes);
        void *out_strides_dev = runtime->malloc_device(stride_bytes);
        void *in_strides_dev = runtime->malloc_device(stride_bytes);
        runtime->memcpy_sync(shape_dev, shape.data(), shape_bytes, LLAISYS_MEMCPY_H2D);
        runtime->memcpy_sync(out_strides_dev, out_strides.data(), stride_bytes, LLAISYS_MEMCPY_H2D);
        runtime->memcpy_sync(in_strides_dev, in_strides.data(), stride_bytes, LLAISYS_MEMCPY_H2D);
        nvidia::rearrange(out->data(), in->data(),
                          reinterpret_cast<const size_t *>(shape_dev),
                          reinterpret_cast<const ptrdiff_t *>(out_strides_dev),
                          reinterpret_cast<const ptrdiff_t *>(in_strides_dev),
                          ndim, elem_size, out->numel());
        runtime->free_device(shape_dev);
        runtime->free_device(out_strides_dev);
        runtime->free_device(in_strides_dev);
        return;
    }
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
