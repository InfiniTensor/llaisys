"""Unit tests for the communication layer API.

Tests the comm API via ctypes: init/destroy, rank/size queries,
and allreduce correctness on a single GPU (nranks=1).

Usage:
    python test_comm_api.py [--device nvidia]
"""

import sys
import os
import ctypes
import argparse
import struct

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import llaisys
from llaisys.libllaisys import LIB_LLAISYS
from llaisys.libllaisys.llaisys_types import llaisysDataType_t, llaisysStream_t


# --- Comm API ctypes bindings ---

# Matches llaisysCommBackend_t
LLAISYS_COMM_NCCL = 0

# Matches llaisysReduceOp_t
LLAISYS_REDUCE_SUM = 0
LLAISYS_REDUCE_MAX = 3

# Matches llaisysDataType_t
LLAISYS_FLOAT32 = 13

llaisysComm_t = ctypes.c_void_p

# comm_init_api: int (*)(llaisysComm_t*, int rank, int size)
comm_init_api = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(llaisysComm_t), ctypes.c_int, ctypes.c_int)
# comm_destroy_api: void (*)(llaisysComm_t)
comm_destroy_api = ctypes.CFUNCTYPE(None, llaisysComm_t)
# comm_get_rank_api: int (*)(llaisysComm_t)
comm_get_rank_api = ctypes.CFUNCTYPE(ctypes.c_int, llaisysComm_t)
# comm_get_size_api: int (*)(llaisysComm_t)
comm_get_size_api = ctypes.CFUNCTYPE(ctypes.c_int, llaisysComm_t)
# comm_allreduce_api: void (*)(const void*, void*, size_t, dtype, op, comm, stream)
comm_allreduce_api = ctypes.CFUNCTYPE(
    None,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
    ctypes.c_int, ctypes.c_int,
    llaisysComm_t, llaisysStream_t,
)


class LlaisysCommAPI(ctypes.Structure):
    _fields_ = [
        ("init", comm_init_api),
        ("destroy", comm_destroy_api),
        ("get_rank", comm_get_rank_api),
        ("get_size", comm_get_size_api),
        ("allreduce", comm_allreduce_api),
        ("broadcast", ctypes.c_void_p),  # skip full typing
        ("send", ctypes.c_void_p),
        ("recv", ctypes.c_void_p),
    ]


def get_comm_api(backend=LLAISYS_COMM_NCCL):
    LIB_LLAISYS.llaisysGetCommAPI.argtypes = [ctypes.c_int]
    LIB_LLAISYS.llaisysGetCommAPI.restype = ctypes.POINTER(LlaisysCommAPI)
    return LIB_LLAISYS.llaisysGetCommAPI(backend).contents


# --- Tests ---

def test_init_destroy(api):
    """Test communicator init and destroy with nranks=1."""
    print("=== test_init_destroy ===")
    comm = llaisysComm_t()
    ret = api.init(ctypes.byref(comm), 0, 1)
    assert ret == 0, f"commInit returned {ret}"
    assert comm.value is not None, "comm handle is null"
    api.destroy(comm)
    print("  PASSED")


def test_rank_size(api):
    """Test rank/size queries on a single-rank communicator."""
    print("=== test_rank_size ===")
    comm = llaisysComm_t()
    ret = api.init(ctypes.byref(comm), 0, 1)
    assert ret == 0

    rank = api.get_rank(comm)
    size = api.get_size(comm)
    assert rank == 0, f"Expected rank 0, got {rank}"
    assert size == 1, f"Expected size 1, got {size}"

    api.destroy(comm)
    print("  PASSED")


def test_allreduce_sum(api, runtime_api):
    """Test allreduce SUM on a single rank (result should equal input)."""
    print("=== test_allreduce_sum ===")
    comm = llaisysComm_t()
    ret = api.init(ctypes.byref(comm), 0, 1)
    assert ret == 0

    stream = runtime_api.create_stream()
    count = 4
    nbytes = count * 4  # float32

    sendbuf = runtime_api.malloc_device(nbytes)
    recvbuf = runtime_api.malloc_device(nbytes)

    # Prepare input: [1.0, 2.0, 3.0, 4.0]
    host_data = struct.pack("ffff", 1.0, 2.0, 3.0, 4.0)
    host_buf = ctypes.create_string_buffer(host_data)
    runtime_api.memcpy_sync(sendbuf, ctypes.cast(host_buf, ctypes.c_void_p).value, nbytes, 1)  # H2D

    api.allreduce(sendbuf, recvbuf, count, LLAISYS_FLOAT32, LLAISYS_REDUCE_SUM, comm, stream)
    runtime_api.stream_synchronize(stream)

    # Copy result back
    out_buf = ctypes.create_string_buffer(nbytes)
    runtime_api.memcpy_sync(ctypes.cast(out_buf, ctypes.c_void_p).value, recvbuf, nbytes, 2)  # D2H

    result = struct.unpack("ffff", out_buf.raw)
    expected = (1.0, 2.0, 3.0, 4.0)
    for i in range(count):
        assert abs(result[i] - expected[i]) < 1e-5, f"Mismatch at [{i}]: {result[i]} != {expected[i]}"

    runtime_api.free_device(sendbuf)
    runtime_api.free_device(recvbuf)
    runtime_api.destroy_stream(stream)
    api.destroy(comm)
    print("  PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="nvidia", choices=["nvidia", "iluvatar"], type=str)
    args = parser.parse_args()

    device_type = llaisys.DeviceType.NVIDIA if args.device == "nvidia" else llaisys.DeviceType.ILUVATAR
    runtime_api = llaisys.RuntimeAPI(device_type)

    api = get_comm_api(LLAISYS_COMM_NCCL)

    test_init_destroy(api)
    test_rank_size(api)
    test_allreduce_sum(api, runtime_api)

    print("\n\033[92mAll comm API tests passed!\033[0m\n")
