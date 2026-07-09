import llaisys
import torch
from test_utils import *
import argparse
import sys


def test_basic_runtime_api(device_name: str = "cpu"):

    from llaisys.libllaisys import LIB_LLAISYS, llaisysDeviceType_t
    from ctypes import c_int

    device_type = llaisys_device(device_name)

    # Check whether this library was compiled with support for the requested device.
    is_supported = LIB_LLAISYS.llaisysIsDeviceSupported(llaisysDeviceType_t(device_type.value))
    if not is_supported:
        raise RuntimeError(
            f"The llaisys library was NOT compiled with {device_name.upper()} support.\n"
            f"  Recompile with GPU support enabled:\n"
            f"    xmake f --nv-gpu=y && xmake -j$(nproc) && xmake install"
        )

    api = llaisys.RuntimeAPI(device_type)
    ndev = api.get_device_count()
    print(f"Found {ndev} {device_name} devices")

    if ndev == 0:
        if device_name == "cpu":
            raise RuntimeError(
                "CPU device count is 0, which is unexpected. "
                "Something is wrong with the runtime."
            )
        else:
            raise RuntimeError(
                f"No {device_name} devices were found (library has {device_name.upper()} support compiled in).\n"
                "  Possible causes:\n"
                "  1. GPU drivers are not installed or the GPU is not accessible in this container.\n"
                "  2. CUDA_VISIBLE_DEVICES is set to empty string (\"\") which hides all GPUs.\n"
                "     Use CUDA_VISIBLE_DEVICES=0 (or unset it) to expose GPUs.\n"
                "  Hint: run 'nvidia-smi' to verify GPU visibility."
            )

    for i in range(ndev):
        print(f"Testing device {i}...")
        api.set_device(i)
        test_memcpy(api, 1024 * 1024)
        print("     Passed")


def test_memcpy(api, size_bytes: int):
    a = torch.zeros((size_bytes,), dtype=torch.uint8, device=torch_device("cpu"))
    b = torch.ones_like(a)
    device_a = api.malloc_device(size_bytes)
    device_b = api.malloc_device(size_bytes)

    # a -> device_a
    api.memcpy_sync(
        device_a,
        a.data_ptr(),
        size_bytes,
        llaisys.MemcpyKind.H2D,
    )
    # device_a -> device_b
    api.memcpy_sync(
        device_b,
        device_a,
        size_bytes,
        llaisys.MemcpyKind.D2D,
    )
    # device_b -> b
    api.memcpy_sync(
        b.data_ptr(),
        device_b,
        size_bytes,
        llaisys.MemcpyKind.D2H,
    )

    api.free_device(device_a)
    api.free_device(device_b)

    torch.testing.assert_close(a, b)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    args = parser.parse_args()
    test_basic_runtime_api(args.device)

    print("\033[92mTest passed!\033[0m\n")
