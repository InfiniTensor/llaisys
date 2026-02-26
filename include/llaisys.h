#ifndef __LLAISYS_H__
#define __LLAISYS_H__

#if defined(_WIN32)
#define __export __declspec(dllexport)
#elif defined(__GNUC__) && ((__GNUC__ >= 4) || (__GNUC__ == 3 && __GNUC_MINOR__ >= 3))
#define __export __attribute__((visibility("default")))
#else
#define __export
#endif

#ifdef __cplusplus
#define __C extern "C"
#include <cstddef>
#include <cstdint>
#else
#define __C
#include <stddef.h>
#include <stdint.h>
#endif

// Device Types
typedef enum {
    LLAISYS_DEVICE_CPU = 0,
    //// TODO: Add more device types here. Numbers need to be consecutive.
    LLAISYS_DEVICE_NVIDIA = 1,
    LLAISYS_DEVICE_TYPE_COUNT
} llaisysDeviceType_t;

// Data Types
typedef enum {
    LLAISYS_DTYPE_INVALID = 0,
    LLAISYS_DTYPE_BYTE = 1,
    LLAISYS_DTYPE_BOOL = 2,
    LLAISYS_DTYPE_I8 = 3,
    LLAISYS_DTYPE_I16 = 4,
    LLAISYS_DTYPE_I32 = 5,
    LLAISYS_DTYPE_I64 = 6,
    LLAISYS_DTYPE_U8 = 7,
    LLAISYS_DTYPE_U16 = 8,
    LLAISYS_DTYPE_U32 = 9,
    LLAISYS_DTYPE_U64 = 10,
    LLAISYS_DTYPE_F8 = 11,
    LLAISYS_DTYPE_F16 = 12,
    LLAISYS_DTYPE_F32 = 13,
    LLAISYS_DTYPE_F64 = 14,
    LLAISYS_DTYPE_C16 = 15,
    LLAISYS_DTYPE_C32 = 16,
    LLAISYS_DTYPE_C64 = 17,
    LLAISYS_DTYPE_C128 = 18,
    LLAISYS_DTYPE_BF16 = 19,
} llaisysDataType_t;

// Runtime Types
// Stream
typedef void *llaisysStream_t;

// Memory Copy Directions
typedef enum {
    LLAISYS_MEMCPY_H2H = 0,
    LLAISYS_MEMCPY_H2D = 1,
    LLAISYS_MEMCPY_D2H = 2,
    LLAISYS_MEMCPY_D2D = 3,
} llaisysMemcpyKind_t;

#endif // __LLAISYS_H__
