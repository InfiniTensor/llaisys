#ifndef LLAISYS_TENSOR_H
#define LLAISYS_TENSOR_H

#include "../llaisys.h"

__C {
    typedef struct LlaisysTensor *llaisysTensor_t;

    __export llaisysTensor_t tensorCreate(
        size_t * shape,
        size_t ndim,
        llaisysDataType_t dtype,
        llaisysDeviceType_t device_type,
        int device_id);

    __export void tensorDestroy(
        llaisysTensor_t tensor);

    __export void *tensorGetData(
        llaisysTensor_t tensor);

    __export size_t tensorGetNdim(
        llaisysTensor_t tensor);

    __export void tensorGetShape(
        llaisysTensor_t tensor,
        size_t * shape);

    __export void tensorGetStrides(
        llaisysTensor_t tensor,
        ptrdiff_t * strides);

    __export llaisysDataType_t tensorGetDataType(
        llaisysTensor_t tensor);

    __export llaisysDeviceType_t tensorGetDeviceType(
        llaisysTensor_t tensor);

    __export int tensorGetDeviceId(
        llaisysTensor_t tensor);

    __export void tensorDebug(
        llaisysTensor_t tensor);

    __export uint8_t tensorIsContiguous(
        llaisysTensor_t tensor);

    __export void tensorLoad(
        llaisysTensor_t tensor,
        const void *data);

    __export llaisysTensor_t tensorView(
        llaisysTensor_t tensor,
        size_t * shape,
        size_t ndim);

    __export llaisysTensor_t tensorPermute(
        llaisysTensor_t tensor,
        size_t * order);

    __export llaisysTensor_t tensorSlice(
        llaisysTensor_t tensor,
        size_t dim,
        size_t start,
        size_t end);
}

#endif // LLAISYS_TENSOR_H
