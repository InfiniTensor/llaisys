import llaisys
import torch
from test_utils import *
import argparse


def test_basic_runtime_api(device_name: str = "cpu"):

    api = llaisys.RuntimeAPI(llaisys_device(device_name))

    ndev = api.get_device_count()
    print(f"Found {ndev} {device_name} devices")
    if ndev == 0:
        print("     Skipped")
        return

    for i in range(ndev):
        print("Testing device {i}...")
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

    torch.testing.assert_close(a, b)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    args = parser.parse_args()
    test_basic_runtime_api(args.device)
    
    print("\033[92mTest passed!\033[0m\n")
