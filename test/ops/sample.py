"""
测试随机采样算子：Temperature、Top-K、Top-P。
不依赖完整模型与 torch，仅验证 Ops.sample 绑定与基本行为。
"""
import sys
import os
import numpy as np
from ctypes import c_void_p

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
import llaisys_py


def _tensor_from_numpy(arr, dtype_llaisys, device=llaisys_py.DeviceType.CPU):
    """从 numpy 数组创建 llaisys Tensor 并拷贝数据（仅 CPU）。"""
    t = llaisys_py.Tensor(arr.shape, dtype=dtype_llaisys, device=device)
    buf = arr.ctypes.data_as(c_void_p)
    nbytes = arr.nbytes
    llaisys_py.RuntimeAPI(device).memcpy_sync(t.data_ptr(), buf, nbytes, llaisys_py.MemcpyKind.H2H)
    return t


def _read_int64(tensor):
    """从 1 元素的 int64 Tensor 读出标量。"""
    out = np.empty(1, dtype=np.int64)
    llaisys_py.RuntimeAPI(llaisys_py.DeviceType.CPU).memcpy_sync(
        out.ctypes.data_as(c_void_p), tensor.data_ptr(), 8, llaisys_py.MemcpyKind.H2H
    )
    return int(out[0])


def test_sample_basic():
    """基本调用：检查返回索引在 [0, voc) 内。"""
    voc = 100
    logits_np = np.random.randn(voc).astype(np.float32)
    logits_ = _tensor_from_numpy(logits_np, llaisys_py.DataType.F32)
    out_idx_ = llaisys_py.Tensor((1,), dtype=llaisys_py.DataType.I64, device=llaisys_py.DeviceType.CPU)
    llaisys_py.Ops.sample(out_idx_, logits_, temperature=1.0, top_k=10, top_p=0.9, seed=42)
    idx = _read_int64(out_idx_)
    assert 0 <= idx < voc, f"sampled index {idx} out of range [0, {voc})"


def test_sample_argmax_like():
    """temperature 接近 0 时应退化为 argmax。"""
    logits_np = np.array([0.1, 0.2, 10.0, 0.3], dtype=np.float32)
    logits_ = _tensor_from_numpy(logits_np, llaisys_py.DataType.F32)
    out_idx_ = llaisys_py.Tensor((1,), dtype=llaisys_py.DataType.I64, device=llaisys_py.DeviceType.CPU)
    llaisys_py.Ops.sample(out_idx_, logits_, temperature=1e-8, top_k=0, top_p=0.0, seed=123)
    idx = _read_int64(out_idx_)
    assert idx == 2, f"expected argmax index 2, got {idx}"


if __name__ == "__main__":
    test_sample_basic()
    test_sample_argmax_like()
    print("\033[92mSample op tests passed!\033[0m\n")
