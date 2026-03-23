"""Worker process for multi-process allreduce test.

Each worker:
1. Rank 0 generates NCCL unique ID and writes to shared file
2. All ranks read the ID file and init communicator
3. Each rank fills sendbuf with (rank+1), runs allreduce SUM
4. Writes result to its result file
"""

import sys
import os
import ctypes
import struct
import time
import argparse

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import llaisys
from llaisys.libllaisys import LIB_LLAISYS
from llaisys.libllaisys.llaisys_types import llaisysStream_t

# Constants
LLAISYS_COMM_NCCL = 0
LLAISYS_REDUCE_SUM = 0
LLAISYS_FLOAT32 = 13
NCCL_UNIQUE_ID_BYTES = 128

llaisysComm_t = ctypes.c_void_p

# Minimal ctypes bindings for comm API
comm_init_api = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(llaisysComm_t), ctypes.c_int, ctypes.c_int)
comm_destroy_api = ctypes.CFUNCTYPE(None, llaisysComm_t)
comm_allreduce_api = ctypes.CFUNCTYPE(
    None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
    ctypes.c_int, ctypes.c_int, llaisysComm_t, llaisysStream_t,
)


class LlaisysCommAPI(ctypes.Structure):
    _fields_ = [
        ("init", comm_init_api),
        ("destroy", comm_destroy_api),
        ("get_rank", ctypes.c_void_p),
        ("get_size", ctypes.c_void_p),
        ("allreduce", comm_allreduce_api),
        ("broadcast", ctypes.c_void_p),
        ("send", ctypes.c_void_p),
        ("recv", ctypes.c_void_p),
    ]


def get_nccl_unique_id():
    """Call ncclGetUniqueId via the NCCL library directly."""
    try:
        nccl = ctypes.CDLL("libnccl.so.2")
    except OSError:
        nccl = ctypes.CDLL("libnccl.so")
    nccl.ncclGetUniqueId.argtypes = [ctypes.POINTER(NcclUniqueId)]
    nccl.ncclGetUniqueId.restype = ctypes.c_int
    uid = NcclUniqueId()
    ret = nccl.ncclGetUniqueId(ctypes.byref(uid))
    assert ret == 0, f"ncclGetUniqueId failed: {ret}"
    return bytes(uid.internal)


class NcclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * NCCL_UNIQUE_ID_BYTES)]


def nccl_comm_init_rank(nranks, uid_bytes, rank):
    """Call ncclCommInitRank directly to pass the shared unique ID."""
    try:
        nccl = ctypes.CDLL("libnccl.so.2")
    except OSError:
        nccl = ctypes.CDLL("libnccl.so")
    nccl.ncclCommInitRank.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, NcclUniqueId, ctypes.c_int]
    nccl.ncclCommInitRank.restype = ctypes.c_int
    comm = ctypes.c_void_p()
    uid = NcclUniqueId()
    ctypes.memmove(uid.internal, uid_bytes, NCCL_UNIQUE_ID_BYTES)
    ret = nccl.ncclCommInitRank(ctypes.byref(comm), nranks, uid, rank)
    assert ret == 0, f"ncclCommInitRank failed: {ret}"
    return comm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--nranks", type=int, required=True)
    parser.add_argument("--device", default="nvidia")
    parser.add_argument("--id_file", required=True)
    parser.add_argument("--result_file", required=True)
    args = parser.parse_args()

    device_type = llaisys.DeviceType.NVIDIA if args.device == "nvidia" else llaisys.DeviceType.ILUVATAR
    runtime_api = llaisys.RuntimeAPI(device_type)
    runtime_api.set_device(0)  # Each process sees one GPU via CUDA_VISIBLE_DEVICES

    # Rank 0 generates and writes unique ID; others wait and read
    if args.rank == 0:
        uid_bytes = get_nccl_unique_id()
        with open(args.id_file, "wb") as f:
            f.write(uid_bytes)
    else:
        for _ in range(100):  # wait up to 10s
            if os.path.exists(args.id_file) and os.path.getsize(args.id_file) >= NCCL_UNIQUE_ID_BYTES:
                break
            time.sleep(0.1)
        with open(args.id_file, "rb") as f:
            uid_bytes = f.read()

    # Init communicator with shared unique ID
    comm = nccl_comm_init_rank(args.nranks, uid_bytes, args.rank)

    # Get comm API for allreduce
    LIB_LLAISYS.llaisysGetCommAPI.argtypes = [ctypes.c_int]
    LIB_LLAISYS.llaisysGetCommAPI.restype = ctypes.POINTER(LlaisysCommAPI)
    api = LIB_LLAISYS.llaisysGetCommAPI(LLAISYS_COMM_NCCL).contents

    stream = runtime_api.create_stream()
    count = 4
    nbytes = count * 4

    sendbuf = runtime_api.malloc_device(nbytes)
    recvbuf = runtime_api.malloc_device(nbytes)

    # Fill sendbuf with (rank + 1)
    val = float(args.rank + 1)
    host_data = struct.pack("ffff", val, val, val, val)
    host_buf = ctypes.create_string_buffer(host_data)
    runtime_api.memcpy_sync(sendbuf, ctypes.cast(host_buf, ctypes.c_void_p).value, nbytes, 1)  # H2D

    # Allreduce SUM using the comm handle we initialized directly
    api.allreduce(sendbuf, recvbuf, count, LLAISYS_FLOAT32, LLAISYS_REDUCE_SUM, comm, stream)
    runtime_api.stream_synchronize(stream)

    # Copy result back to host and write to file
    out_buf = ctypes.create_string_buffer(nbytes)
    runtime_api.memcpy_sync(ctypes.cast(out_buf, ctypes.c_void_p).value, recvbuf, nbytes, 2)  # D2H

    with open(args.result_file, "wb") as f:
        f.write(out_buf.raw)

    runtime_api.free_device(sendbuf)
    runtime_api.free_device(recvbuf)
    runtime_api.destroy_stream(stream)

    # Destroy comm via NCCL directly
    try:
        nccl = ctypes.CDLL("libnccl.so.2")
    except OSError:
        nccl = ctypes.CDLL("libnccl.so")
    nccl.ncclCommDestroy(comm)


if __name__ == "__main__":
    main()
