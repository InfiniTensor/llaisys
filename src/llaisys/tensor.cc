#include "llaisys_tensor.hpp"

#include <vector>

__C {
    llaisysTensor_t tensorCreate(
        size_t * shape,
        size_t ndim,
        llaisysDataType_t dtype,
        llaisysDeviceType_t device_type,
        int device_id) {
        std::vector<size_t> shape_vec(shape, shape + ndim);
        return new LlaisysTensor{llaisys::Tensor::create(shape_vec, dtype, device_type, device_id)};
    }

    void tensorDestroy(
        llaisysTensor_t tensor) {
        delete tensor;
    }

    void *tensorGetData(
        llaisysTensor_t tensor) {
        return tensor->tensor->data();
    }

    size_t tensorGetNdim(
        llaisysTensor_t tensor) {
        return tensor->tensor->ndim();
    }

    void tensorGetShape(
        llaisysTensor_t tensor,
        size_t * shape) {
        std::copy(tensor->tensor->shape().begin(), tensor->tensor->shape().end(), shape);
    }

    void tensorGetStrides(
        llaisysTensor_t tensor,
        ptrdiff_t * strides) {
        std::copy(tensor->tensor->strides().begin(), tensor->tensor->strides().end(), strides);
    }

    llaisysDataType_t tensorGetDataType(
        llaisysTensor_t tensor) {
        return tensor->tensor->dtype();
    }

    llaisysDeviceType_t tensorGetDeviceType(
        llaisysTensor_t tensor) {
        return tensor->tensor->deviceType();
    }

    int tensorGetDeviceId(
        llaisysTensor_t tensor) {
        return tensor->tensor->deviceId();
    }

    void tensorDebug(
        llaisysTensor_t tensor) {
        tensor->tensor->debug();
    }

    uint8_t tensorIsContiguous(
        llaisysTensor_t tensor) {
        return uint8_t(tensor->tensor->isContiguous());
    }

    void tensorLoad(
        llaisysTensor_t tensor,
        const void *data) {
        tensor->tensor->load(data);
    }

    llaisysTensor_t tensorView(
        llaisysTensor_t tensor,
        size_t * shape,
        size_t ndim) {
        std::vector<size_t> shape_vec(shape, shape + ndim);
        return new LlaisysTensor{tensor->tensor->view(shape_vec)};
    }

    llaisysTensor_t tensorPermute(
        llaisysTensor_t tensor,
        size_t * order) {
        std::vector<size_t> order_vec(order, order + tensor->tensor->ndim());
        return new LlaisysTensor{tensor->tensor->permute(order_vec)};
    }

    llaisysTensor_t tensorSlice(
        llaisysTensor_t tensor,
        size_t dim,
        size_t start,
        size_t end) {
        return new LlaisysTensor{tensor->tensor->slice(dim, start, end)};
    }
}
