import sys
import os
import torch
import ctypes

# Add test root to path if needed, though we use direct implementation here
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import llaisys
from llaisys import Tensor, Ops, DataType, DeviceType

def create_llaisys_tensor(data, dtype=DataType.F32):
    # Create from torch tensor
    t = torch.tensor(data)
    if dtype == DataType.F32:
        t = t.float()
    elif dtype == DataType.I64:
        t = t.long()
    
    ls_t = Tensor(t.shape, dtype=dtype, device=DeviceType.CPU)
    ls_t.load(t.data_ptr())
    return ls_t

def get_item(tensor):
    # Read single scalar value from CPU tensor
    assert tensor.ndim() == 1 and tensor.shape()[0] == 1
    assert tensor.device_type() == DeviceType.CPU
    ptr = tensor.data_ptr()
    if tensor.dtype() == DataType.I64:
        return ctypes.cast(ptr, ctypes.POINTER(ctypes.c_int64)).contents.value
    elif tensor.dtype() == DataType.F32:
        return ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float)).contents.value
    return None

def test_sample_topk_1():
    """Test if top_k=1 is equivalent to argmax"""
    logits = create_llaisys_tensor([0.1, 0.2, 0.7, 0.0], DataType.F32)
    # Output tensor must be allocated with correct shape
    out_idx = Tensor((1,), dtype=DataType.I64, device=DeviceType.CPU)
    
    # Top-K = 1, temperature=1.0
    Ops.sample(out_idx, logits, top_k=1, temperature=1.0, seed=42)
    val = get_item(out_idx)
    assert val == 2, f"Expected index 2, got {val}"

def test_sample_seed():
    """Test reproducibility with fixed seed"""
    logits = create_llaisys_tensor([0.3, 0.3, 0.3, 0.1], DataType.F32)
    out1 = Tensor((1,), dtype=DataType.I64, device=DeviceType.CPU)
    out2 = Tensor((1,), dtype=DataType.I64, device=DeviceType.CPU)
    
    Ops.sample(out1, logits, temperature=1.0, seed=12345)
    Ops.sample(out2, logits, temperature=1.0, seed=12345)
    
    val1 = get_item(out1)
    val2 = get_item(out2)
    assert val1 == val2, f"Results should be identical with same seed: {val1} vs {val2}"

def test_sample_temperature_small():
    """Test if small temperature tends to argmax"""
    logits = create_llaisys_tensor([0.4, 0.5, 0.1], DataType.F32)
    out_idx = Tensor((1,), dtype=DataType.I64, device=DeviceType.CPU)
    
    # Small temperature -> sharper probability distribution -> max value
    Ops.sample(out_idx, logits, temperature=1e-5, seed=42)
    val = get_item(out_idx)
    assert val == 1, f"Expected index 1, got {val}"

if __name__ == "__main__":
    try:
        test_sample_topk_1()
        test_sample_seed()
        test_sample_temperature_small()
        print("test_sample.py passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
