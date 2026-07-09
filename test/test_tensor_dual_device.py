import argparse

import llaisys
import torch

from test_utils import check_equal, llaisys_device


def test_tensor_to_dual_device(device_id: int = 0):
    api = llaisys.RuntimeAPI(llaisys_device("nvidia"))
    if api.get_device_count() == 0:
        print("===Test tensor.to dual-device===")
        print("No nvidia devices found, skipped")
        return

    if device_id < 0 or device_id >= api.get_device_count():
        raise ValueError(f"Invalid nvidia device_id={device_id}, count={api.get_device_count()}")

    print("===Test tensor.to dual-device===")
    torch_tensor_cpu = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    torch_tensor_gpu = torch_tensor_cpu.to(torch.device(f"cuda:{device_id}"))

    cpu_tensor = llaisys.Tensor(
        (4, 8), dtype=llaisys.DataType.F32, device=llaisys.DeviceType.CPU
    )
    cpu_tensor.load(torch_tensor_cpu.data_ptr())

    gpu_tensor = cpu_tensor.to(llaisys.DeviceType.NVIDIA, device_id)
    assert gpu_tensor.device_type() == llaisys.DeviceType.NVIDIA
    assert gpu_tensor.device_id() == device_id
    assert check_equal(gpu_tensor, torch_tensor_gpu)

    cpu_back = gpu_tensor.to(llaisys.DeviceType.CPU, 0)
    assert cpu_back.device_type() == llaisys.DeviceType.CPU
    assert check_equal(cpu_back, torch_tensor_cpu)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", default=0, type=int)
    args = parser.parse_args()

    test_tensor_to_dual_device(args.device_id)
    print("\n\033[92mTest passed!\033[0m\n")
