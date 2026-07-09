import sys
import os
import torch
import numpy as np

# Adjust path to find llaisys package
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../python"))
sys.path.insert(0, parent_dir)
import llaisys

def torch_rope(y: torch.Tensor, x: torch.Tensor, pos_ids: torch.Tensor, theta: float):
    seq_len, n_heads, head_dim = y.shape
    x_a, x_b = x[..., : head_dim // 2], x[..., head_dim // 2 :]
    positions = pos_ids.to(torch.float32).unsqueeze(1)
    i = torch.arange(0, head_dim // 2, dtype=torch.float32, device=y.device)
    freqs = positions / (theta ** (2 * i / head_dim))
    sin, cos = freqs.sin(), freqs.cos()
    sin = sin.unsqueeze(1)
    cos = cos.unsqueeze(1)
    y[..., : head_dim // 2] = x_a * cos - x_b * sin
    y[..., head_dim // 2 :] = x_b * cos + x_a * sin

def debug_rope():
    # Configuration matching the failing case
    shape = (512, 4, 4096)
    start_pos = 512
    end_pos = 1024 
    dtype = torch.float32
    theta = 10000.0

    print(f"Debugging RoPE with shape={shape}, range=[{start_pos}, {end_pos}), dtype={dtype}")

    # 1. Setup Data
    torch.manual_seed(42)
    x = torch.randn(shape, dtype=dtype)
    pos_ids = torch.arange(start_pos, end_pos, dtype=torch.int64)
    y_torch = torch.zeros_like(x)

    # 2. Run PyTorch
    torch_rope(y_torch, x, pos_ids, theta)

    # 3. Setup LLAISYS
    # Helpers
    device_enum = llaisys.DeviceType.CPU
    dt_enum = llaisys.DataType.F32
    api = llaisys.RuntimeAPI(device_enum)
    
    # Create LLAISYS tensors
    x_ll = llaisys.Tensor(shape, dtype=dt_enum, device=device_enum)
    y_ll = llaisys.Tensor(shape, dtype=dt_enum, device=device_enum)
    pos_ll = llaisys.Tensor((len(pos_ids),), dtype=llaisys.DataType.I64, device=device_enum)

    # Copy Input Data (x, pos_ids)
    # Using HostToHost since we are on CPU
    kind = llaisys.MemcpyKind.HostToHost 
    
    api.memcpy_sync(x_ll.data_ptr(), x.data_ptr(), x.numel() * x.element_size(), kind)
    api.memcpy_sync(pos_ll.data_ptr(), pos_ids.data_ptr(), pos_ids.numel() * pos_ids.element_size(), kind)
    
    # Run Op
    llaisys.Ops.rope(y_ll, x_ll, pos_ll, theta)

    # Copy Output Data back
    y_llaisys = torch.zeros_like(x)
    api.memcpy_sync(y_llaisys.data_ptr(), y_ll.data_ptr(), y_ll.numel() * y_ll.element_size(), kind)

    # 4. Analyze Error
    diff = (y_torch - y_llaisys).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"Max Diff: {max_diff:.2e}")
    print(f"Mean Diff: {mean_diff:.2e}")

    # 5. Detailed Breakdown
    if max_diff > 1e-5: # Only show details if significant error
        max_indices = torch.nonzero(diff == max_diff)
        if len(max_indices) > 0:
            idx = max_indices[0]
            seq_idx, head_idx, dim_idx = idx.tolist()
            print(f"Max error at index: seq={seq_idx}, head={head_idx}, dim={dim_idx}")
            curr_pos = pos_ids[seq_idx].item()
            print(f"Pos ID at failure: {curr_pos}")
            
            # Theoretical calc
            head_dim = shape[2]
            freq_idx = dim_idx if dim_idx < head_dim // 2 else dim_idx - head_dim // 2
            
            freq_exponent_f = (2.0 * freq_idx) / head_dim
            denom_f = theta ** freq_exponent_f
            angle_f = curr_pos / denom_f
            
            # Double precision check
            freq_exponent_d = (2.0 * freq_idx) / float(head_dim)
            denom_d = theta ** freq_exponent_d
            angle_d = curr_pos / denom_d
            
            print(f"Angle(float) approx: {angle_f}")
            print(f"Angle(double) approx: {angle_d}")
            
            val_t = y_torch[seq_idx, head_idx, dim_idx].item()
            val_l = y_llaisys[seq_idx, head_idx, dim_idx].item()
            print(f"Values: Torch={val_t:.8f}, LLAISYS={val_l:.8f}")
            print(f"Diff: {abs(val_t - val_l):.8f}")

    if max_diff > 5e-4:
        print("\n\033[91mFAILED: Error exceeds 5e-4\033[0m")
        sys.exit(1)
    else:
        print("\n\033[92mPASSED\033[0m")

if __name__ == "__main__":
    debug_rope()
