import llaisys

import torch
from test_utils import *
import argparse


def test_tensor():
    torch_tensor = torch.arange(60, dtype=torch_dtype("i64")).reshape(3, 4, 5)
    llaisys_tensor = llaisys.Tensor(
        (3, 4, 5), dtype=llaisys_dtype("i64"), device=llaisys_device("cpu")
    )

    # Test load
    print("===Test load===")
    llaisys_tensor.load(torch_tensor.data_ptr())
    llaisys_tensor.debug()
    assert llaisys_tensor.is_contiguous() == torch_tensor.is_contiguous()
    assert check_equal(llaisys_tensor, torch_tensor)

    # Test view
    print("===Test view===")
    torch_tensor_view = torch_tensor.view(6, 10)
    llaisys_tensor_view = llaisys_tensor.view(6, 10)
    llaisys_tensor_view.debug()
    assert llaisys_tensor_view.shape() == torch_tensor_view.shape
    assert llaisys_tensor_view.strides() == torch_tensor_view.stride()
    assert llaisys_tensor.is_contiguous() == torch_tensor.is_contiguous()
    assert check_equal(llaisys_tensor_view, torch_tensor_view)

    # Test permute
    print("===Test permute===")
    torch_tensor_perm = torch_tensor.permute(2, 0, 1)
    llaisys_tensor_perm = llaisys_tensor.permute(2, 0, 1)
    llaisys_tensor_perm.debug()
    assert llaisys_tensor_perm.shape() == torch_tensor_perm.shape
    assert llaisys_tensor_perm.strides() == torch_tensor_perm.stride()
    assert llaisys_tensor.is_contiguous() == torch_tensor.is_contiguous()
    assert check_equal(llaisys_tensor_perm, torch_tensor_perm)

    # Test slice
    print("===Test slice===")
    torch_tensor_slice = torch_tensor[:, :, 1:4]
    llaisys_tensor_slice = llaisys_tensor.slice(2, 1, 4)
    llaisys_tensor_slice.debug()
    assert llaisys_tensor_slice.shape() == torch_tensor_slice.shape
    assert llaisys_tensor_slice.strides() == torch_tensor_slice.stride()
    assert llaisys_tensor.is_contiguous() == torch_tensor.is_contiguous()
    assert check_equal(llaisys_tensor_slice, torch_tensor_slice)


if __name__ == "__main__":
    test_tensor()

    print("\n\033[92mTest passed!\033[0m\n")
