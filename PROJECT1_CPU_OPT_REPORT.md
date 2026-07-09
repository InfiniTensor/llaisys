# 项目#1 CPU 推理优化笔记（LLAISYS）

## 1. 一页结论
- 优化对象：`Ops.linear`（CPU 上最耗时，等价 GEMM）。
- 主线策略：`朴素循环 -> SIMD+OpenMP -> OpenBLAS`。
- 结果：`f32` 从 `6173.788 ms` 降到 `253.513 ms`，累计约 `24.35x`。
- 扩展：`f16/bf16` 已接入专用快速路径，不再在最内层循环频繁类型转换。

## 2. 测试口径（统一）
- 平台：Windows x64, MSVC 19.50。
- 代码：`llaisys`。
- Python：`./.venv`。
- 线程：`OMP_NUM_THREADS=8`。
- 形状：`x=(512,4096)`, `w=(4096,4096)`, `bias=(4096,)`, `out=(512,4096)`。
- benchmark：`warmup=1`, `repeat=3`。
- 说明：Baseline 为历史记录；其余为同机复测。

## 3. 里程碑与改动
### A. Baseline（优化前）
- 三重循环逐元素乘加。
- 无 SIMD，无 BLAS。

### B. SIMD + OpenMP
- 文件：`src/ops/linear/op.cpp`。
- `f32` 点积内核：`_mm256_loadu_ps` + `_mm256_fmadd_ps`。
- 外层 `m` 维并行：`#pragma omp parallel for`。
- 不支持 AVX2 时自动回退标量路径。

### C. OpenBLAS
- 文件：`src/ops/linear/op.cpp`。
- `ENABLE_OPENBLAS` 下，`f32` 调用 `cblas_sgemm` 完成主计算，再加 bias。
- 文件：`xmake.lua`、`xmake/cpu.lua`。
- 新增构建开关：`--openblas=y|n`。

### D. `f16/bf16` 专用高性能路径（已完成）
- 文件：`src/ops/linear/op.cpp`。
- 新增 `linear_impl_lowp_fast<T>`。
- `weight/bias` 先批量转 `float`，避免最内层重复 `cast`。
- OpenBLAS 开启时：`in` 也批量转 `float` 后走 `sgemm`。
- OpenBLAS 关闭时：每线程复用 `in_row_f`，继续复用 `dot_f32`。

## 4. 性能对比
### 4.1 `f32` 分阶段结果
| 阶段 | LLAISYS (ms) | 说明 |
|---|---:|---|
| A. Baseline（历史） | 6173.788 | 优化前 |
| B. SIMD + OpenMP（复测） | 266.430 | `--openblas=n --openmp=y --cpu-avx2=y` |
| C. OpenBLAS（复测） | 253.513 | `--openblas=y --openmp=y --cpu-avx2=y` |

对应 Torch（同测参考）：
- B 阶段 Torch：`56.534 ms`
- C 阶段 Torch：`47.332 ms`

### 4.2 `f16/bf16` 专用路径结果
| dtype | Torch (ms) | LLAISYS (ms) |
|---|---:|---:|
| `f16` | 283.002 | 271.528 |
| `bf16` | 297.637 | 268.177 |

## 5. 加速比（`f32`）
- A -> B：`6173.788 / 266.430 = 23.17x`
- A -> C：`6173.788 / 253.513 = 24.35x`
- B -> C：`266.430 / 253.513 = 1.05x`（约 `5.1%`）

结论：
- 最大收益来自结构性优化（SIMD + 并行）。
- OpenBLAS 在其上继续带来稳定增益。

## 6. 复现实验命令
### 6.1 `f32`：SIMD + OpenMP（关闭 OpenBLAS）
```powershell
xmake f --openblas=n --openmp=y --cpu-avx2=y -y
xmake -y
xmake install -y
$env:OMP_NUM_THREADS=8
D:/86188/大模型学习/llaisys/.venv/Scripts/python.exe -c "import sys; sys.path.insert(0,'test'); import llaisys, torch; from test_utils import random_tensor, benchmark; x,x_=random_tensor((512,4096),'f32','cpu',scale=0.1); w,w_=random_tensor((4096,4096),'f32','cpu',scale=0.01); b,b_=random_tensor((4096,),'f32','cpu'); out,out_=random_tensor((512,4096),'f32','cpu'); f1=lambda: torch.nn.functional.linear(x,w,b,out=out); f2=lambda: llaisys.Ops.linear(out_,x_,w_,b_); benchmark(f1,f2,'cpu',warmup=1,repeat=3)"
```

### 6.2 `f32`：OpenBLAS
```powershell
xmake f --openblas=y --openmp=y --cpu-avx2=y -y
xmake -y
xmake install -y
$env:OMP_NUM_THREADS=8
D:/86188/大模型学习/llaisys/.venv/Scripts/python.exe -c "import sys; sys.path.insert(0,'test'); import llaisys, torch; from test_utils import random_tensor, benchmark; x,x_=random_tensor((512,4096),'f32','cpu',scale=0.1); w,w_=random_tensor((4096,4096),'f32','cpu',scale=0.01); b,b_=random_tensor((4096,),'f32','cpu'); out,out_=random_tensor((512,4096),'f32','cpu'); f1=lambda: torch.nn.functional.linear(x,w,b,out=out); f2=lambda: llaisys.Ops.linear(out_,x_,w_,b_); benchmark(f1,f2,'cpu',warmup=1,repeat=3)"
```
