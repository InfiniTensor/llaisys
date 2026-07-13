import argparse
import os
import tempfile
import uuid

import llaisys
import torch
import torch.multiprocessing as mp


def tensor_from_torch(torch_tensor: torch.Tensor, rank: int) -> llaisys.Tensor:
    tensor = llaisys.Tensor(
        torch_tensor.shape,
        dtype=llaisys.DataType.F32,
        device=llaisys.DeviceType.NVIDIA,
        device_id=rank,
    )
    api = llaisys.RuntimeAPI(llaisys.DeviceType.NVIDIA)
    api.set_device(rank)
    api.memcpy_sync(
        tensor.data_ptr(),
        torch_tensor.data_ptr(),
        torch_tensor.numel() * torch_tensor.element_size(),
        llaisys.MemcpyKind.D2D,
    )
    return tensor


def torch_from_tensor(tensor: llaisys.Tensor, shape, rank: int) -> torch.Tensor:
    result = torch.empty(shape, dtype=torch.float32, device=torch.device(f"cuda:{rank}"))
    api = llaisys.RuntimeAPI(llaisys.DeviceType.NVIDIA)
    api.set_device(rank)
    api.memcpy_sync(
        result.data_ptr(),
        tensor.data_ptr(),
        result.numel() * result.element_size(),
        llaisys.MemcpyKind.D2D,
    )
    return result


def worker(rank: int, world_size: int) -> None:
    api = llaisys.RuntimeAPI(llaisys.DeviceType.NVIDIA)
    api.set_device(rank)
    dist = llaisys.DistributedContext()
    dist.init(rank, world_size)

    local = torch.full((4,), float(rank + 1), dtype=torch.float32, device=torch.device(f"cuda:{rank}"))
    reduced = tensor_from_torch(local.clone(), rank)
    dist.all_reduce(reduced)
    reduced_result = torch_from_tensor(reduced, local.shape, rank)
    expected_reduce = torch.full_like(local, float(world_size * (world_size + 1) // 2))
    torch.testing.assert_close(reduced_result, expected_reduce)

    gathered = dist.all_gather(tensor_from_torch(local, rank))
    gathered_result = torch_from_tensor(gathered, (world_size, 4), rank)
    expected_gather = torch.stack(
        [
            torch.full((4,), float(peer_rank + 1), dtype=torch.float32, device=torch.device(f"cuda:{rank}"))
            for peer_rank in range(world_size)
        ]
    )
    torch.testing.assert_close(gathered_result, expected_gather)

    broadcast_source = (
        torch.tensor([10.0, 11.0, 12.0, 13.0], dtype=torch.float32, device=torch.device(f"cuda:{rank}"))
        if rank == 0
        else torch.zeros((4,), dtype=torch.float32, device=torch.device(f"cuda:{rank}"))
    )
    broadcast_tensor = tensor_from_torch(broadcast_source, rank)
    dist.broadcast(broadcast_tensor, root=0)
    broadcast_result = torch_from_tensor(broadcast_tensor, (4,), rank)
    expected_broadcast = torch.tensor([10.0, 11.0, 12.0, 13.0], dtype=torch.float32, device=torch.device(f"cuda:{rank}"))
    torch.testing.assert_close(broadcast_result, expected_broadcast)

    dist.barrier()
    dist.finalize()
    api.device_synchronize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="nvidia", choices=["nvidia"], type=str)
    parser.add_argument("--world-size", default=8, type=int)
    args = parser.parse_args()

    assert args.device == "nvidia"
    assert torch.cuda.device_count() >= args.world_size

    os.environ["LLAISYS_DIST_BOOTSTRAP_PATH"] = os.path.join(
        tempfile.gettempdir(), f"llaisys_nccl_{uuid.uuid4().hex}.bin"
    )
    mp.spawn(worker, args=(args.world_size,), nprocs=args.world_size, join=True)

    print("\033[92mDistributed test passed!\033[0m\n")
