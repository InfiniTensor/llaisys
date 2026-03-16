#!/usr/bin/env python3
"""
项目#5 张量并行推理测试。

用法（需先 xmake f --nv-gpu=y --nccl=y && xmake && xmake install，并安装 libnccl）：
  .venv/bin/python test/test_tensor_parallel.py --model /path/to/model

需要至少 2 张 GPU；通过 CUDA_VISIBLE_DEVICES 指定每进程使用的卡。
若子进程报 undefined symbol: ncclCommShrink，多为本机 libnccl 版本低于 PyTorch 所需，
可升级 libnccl 或使用 LD_PRELOAD=/path/to/newer/libnccl.so 指定与 PyTorch 匹配的 NCCL。
"""
import argparse
import os
import sys
import tempfile

# 确保能 import 到项目包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llaisys_py.libllaisys import LIB_LLAISYS
from llaisys_py.libllaisys import nccl_comm
from llaisys_py.models.qwen2 import Qwen2
from llaisys_py.libllaisys import DeviceType


def _run_rank(rank: int, world_size: int, model_path: str, unique_id_path: str):
    # 必须在 import llaisys 之前限制本进程可见的 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    # 子进程不 import torch，避免 libtorch_cuda 依赖 ncclCommShrink 与系统 libnccl 版本冲突；Qwen2 在无 torch 时用纯 Python 解析 bf16 权重
    from llaisys_py.libllaisys import LIB_LLAISYS
    from llaisys_py.libllaisys import nccl_comm

    if rank == 0:
        try:
            uid = nccl_comm.get_unique_id(LIB_LLAISYS)
        except AttributeError:
            print("NCCL not compiled (missing llaisysNcclGetUniqueId). Build with --nccl=y.", file=sys.stderr)
            return 1
        with open(unique_id_path, "wb") as f:
            f.write(uid)
            f.flush()
            os.fsync(f.fileno())
    else:
        import time
        for _ in range(50):
            if os.path.isfile(unique_id_path) and os.path.getsize(unique_id_path) >= nccl_comm.LLAISYS_NCCL_UNIQUE_ID_BYTES:
                break
            time.sleep(0.1)
        if not os.path.isfile(unique_id_path) or os.path.getsize(unique_id_path) < nccl_comm.LLAISYS_NCCL_UNIQUE_ID_BYTES:
            print("Rank 1: unique id file not ready or too small", file=sys.stderr)
            return 1
        with open(unique_id_path, "rb") as f:
            uid = f.read()

    if nccl_comm.init_rank(LIB_LLAISYS, rank, world_size, uid) != 0:
        err = nccl_comm.get_last_error(LIB_LLAISYS)
        print(f"Rank {rank}: nccl init failed — {err}", file=sys.stderr)
        return 1

    model = Qwen2(model_path, device=DeviceType.NVIDIA, tp_rank=rank, tp_world_size=world_size)
    prompt = "Hello"
    out = model.generate(
        prompt,
        max_new_tokens=4,
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        seed=42,
    )
    print(f"Rank {rank} output: {out}")
    nccl_comm.destroy(LIB_LLAISYS)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Test tensor-parallel inference (Project #5)")
    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument("--world-size", type=int, default=2, help="Number of TP ranks (default 2)")
    args = parser.parse_args()

    if not os.path.isdir(args.model):
        print(f"Model path not found: {args.model}", file=sys.stderr)
        return 1

    try:
        has_nccl = hasattr(LIB_LLAISYS, "llaisysNcclGetUniqueId")
    except Exception:
        has_nccl = False
    if not has_nccl:
        print("NCCL symbols not found. Rebuild with: xmake f --nv-gpu=y --nccl=y && xmake", file=sys.stderr)
        return 1

    with tempfile.NamedTemporaryFile(delete=False, suffix=".nccl_id") as f:
        unique_id_path = f.name
    try:
        import multiprocessing as mp
        # 必须用 spawn：fork 后子进程继承父进程 CUDA 状态，NCCL/CUDA 不支持在 fork 后使用
        ctx = mp.get_context("spawn")
        procs = []
        for r in range(args.world_size):
            p = ctx.Process(target=_run_rank, args=(r, args.world_size, args.model, unique_id_path))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
        ok = all(p.exitcode == 0 for p in procs)
    finally:
        if os.path.isfile(unique_id_path):
            os.unlink(unique_id_path)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
