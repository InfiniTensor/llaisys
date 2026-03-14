# LLAISYS 项目 2 第二平台（MetaX/MACA）实现报告

## 1. 提交结论

本次提交的目标是把项目 2 的第二平台从“设计稿”推进到“沐曦平台可实际测试”的状态。当前仓库已经完成以下交付：

- 保留原有 `CPU` 与 `NVIDIA` 设备路径，不改坏已有接口和构建开关。
- 新增独立 `METAX` 设备类型，不把沐曦实现塞进 `nvidia` 分支里。
- 打通 MetaX/MACA 的 runtime、资源管理、算子调度和 Python 测试入口。
- 在真实沐曦机器上完成 `runtime -> ops -> infer` 的顺序验证。

当前分支的真实可验证结论是：`CPU + NVIDIA` 路径仍保留，`MetaX/MACA` 已经从设计说明落地为可编译、可运行、可测试的第二平台。

## 2. 平台识别与兼容性判断


### 2.1 本机环境识别结果

在实际验证机器上确认到如下环境：

- `mx-smi 2.2.9`
- `MetaX C500`
- `MACA 3.2.1.10`
- 驱动版本 `3.0.11`
- 编译器 `mxcc 1.0.0`
- Python `3.10.10`
- PyTorch `2.6.0+metax3.2.1.3`
- xmake `v2.8.7+20240401`

本机 SDK 与运行库位置：

- 头文件目录：`/opt/maca/include`
- MACA 运行库目录：`/opt/maca/lib`
- 驱动运行库目录：`/opt/mxdriver/lib`

### 2.2 CUDA 兼容性判断

结论分两层：

- C/C++ SDK 层面不是 CUDA drop-in 兼容，必须单独适配。
- Python / PyTorch 使用层面保留了 CUDA 命名空间语义，可以继续走 `torch.cuda`。

判断依据：

- 本机没有可直接复用的 `nvcc`、`nvidia-smi`、`libcudart`、`libcublas` 运行时路径。
- MetaX 的 C/C++ 接口来自 `<mcr/mc_runtime.h>` 与 `<mcblas/mcblas.h>`，不是 `<cuda_runtime.h>` 与 `<cublas_v2.h>`。
- `torch.cuda.is_available()` 在本机返回 `True`，并且设备名能显示为 `MetaX C500`，说明 PyTorch 层做了 CUDA 语义兼容。

因此，本次实现采取的策略是：

- LLAISYS C/C++ 后端新增独立 `METAX` 分支，单独对接 `mc*` / `mcblas*`。
- Python 测试对照仍复用 `torch.cuda`，不额外重写 Hugging Face 推理逻辑。

## 3. 实现说明

### 3.1 设备抽象与构建

本次改动先扩展设备抽象，再接入平台构建：

- 在 `llaisysDeviceType_t` 中新增 `LLAISYS_DEVICE_METAX`
- 保留原 `CPU` 与 `NVIDIA` 枚举值和分发逻辑
- 在 `xmake.lua` 中新增 `--metax-gpu=y`
- 新增 `xmake/metax.lua`，统一使用 `mxcc` 编译 `.cu`
- 共享库链接时单独增加 MetaX link stub，避免设备侧对象未被最终链接

这样处理的好处是：

- CPU 与 NVIDIA 路径的构建选项保持独立
- 第二平台出问题时不会污染原 NVIDIA 代码路径
- 后续如果需要继续接第三个平台，可以直接沿用同一抽象方式

### 3.2 Runtime 映射

MetaX runtime 完全按 LLAISYS 现有 runtime 抽象接入，主要映射如下：

| LLAISYS runtime | MACA 接口 |
| --- | --- |
| `get_device_count` | `mcGetDeviceCount` |
| `set_device` | `mcSetDevice` |
| `device_synchronize` | `mcDeviceSynchronize` |
| `create_stream` | `mcStreamCreateWithFlags(..., mcStreamNonBlocking)` |
| `destroy_stream` | `mcStreamDestroy` |
| `stream_synchronize` | `mcStreamSynchronize` |
| `malloc_device` | `mcMalloc` |
| `free_device` | `mcFree` |
| `malloc_host` | `mcMallocHost` |
| `free_host` | `mcFreeHost` |
| `memcpy_sync` | `mcMemcpy` |
| `memcpy_async` | `mcMemcpyAsync` |

实现细节：

- 错误处理统一转成 `mcGetErrorString` / `mcblasGetStatusString`
- 关键桥接代码补了中文注释，说明 CUDA 与 MACA 语义差异
- `mcblasCreate` 前显式保证已经 `mcSetDevice`
- `mcblas` handle 按线程和设备缓存，并在每次调用前更新 stream

### 3.3 算子策略

MetaX 目录独立放在：

- `src/device/metax/`
- `src/ops/metax/`

当前算子策略如下：

| 算子 | 当前策略 |
| --- | --- |
| `add` | MetaX kernel |
| `embedding` | MetaX kernel |
| `linear` | `mcblasGemmEx` + bias kernel |
| `rms_norm` | MetaX kernel |
| `rope` | MetaX kernel |
| `swiglu` | MetaX kernel |
| `argmax` | host fallback |
| `self_attention` | host fallback |

其中 `linear` 是本次最关键的部分：

- dtype 映射使用 `MACA_R_32F` / `MACA_R_16F` / `MACA_R_16BF`
- 计算类型以 `MCBLAS_COMPUTE_32F` 为主
- 对 `float32` 路径额外切到 `MCBLAS_COMPUTE_32F_PEDANTIC`
- 数学模式固定为 `MCBLAS_PEDANTIC_MATH`

这样做的原因是 MetaX 与 PyTorch 对照时会有轻微数值漂移，`pedantic` 模式更容易稳定通过当前严格测试。

### 3.4 Python 与测试桥接

Python 层同步做了三件事：

- `DeviceType` 增加 `METAX`
- `test/test_runtime.py`、`test/test_ops.py`、`test/test_infer.py` 增加 `--device metax`
- `test/test_utils.py` 中把 MetaX 映射到 `llaisys.DeviceType.METAX`

同时保留一个重要约束：

- `torch_device("metax")` 仍然返回 `torch.device("cuda:N")`

原因是本机 MetaX PyTorch 并没有暴露一个新的 `torch.device("metax")` 命名空间，而是直接复用了 CUDA 语义。

### 3.5 推理模型支持现状

当前分支已经在推理链路上实际验证的是 `Qwen2` 路径。为了保证文档与仓库一致，本报告不再声称当前分支已经完整交付 `TinyLlama/Llama` 作业支持。

本次实际用于严格推理验证的模型是：

- `trl-internal-testing/tiny-Qwen2ForCausalLM-2.5`

选择这个模型的原因是：

- 体积小，适合当前机器快速做严格一致性测试
- 仓库本地没有现成可直接用于提交验证的模型目录
- 其 `model_type` 为 `qwen2`，与当前分支已验证的模型实现一致

## 4. 构建与实测结果

### 4.1 构建命令

本机以 root 身份运行，因此 `xmake` 需要带 `XMAKE_ROOT=y`：

```bash
XMAKE_ROOT=y xmake f --metax-gpu=y -cv
XMAKE_ROOT=y xmake -r
XMAKE_ROOT=y xmake install
```

以上命令已通过。

### 4.2 CPU 基线验证

以下 CPU 命令已在当前仓库通过：

```bash
python test/test_tensor.py
python test/test_runtime.py --device cpu
python test/test_ops.py --device cpu
python test/test_infer.py --device cpu --test --model_id trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 --prompt hi --max_steps 1
```

说明：

- 这部分用于证明 MetaX 接入没有破坏 CPU 基线路径
- `test_infer.py --device cpu` 已完成 token 级严格一致性校验

### 4.3 MetaX 验证

以下 MetaX 命令已在真实沐曦机器通过：

```bash
python test/test_runtime.py --device metax
python test/test_ops.py --device metax
python test/test_infer.py --device metax --test --model_id trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 --prompt hi --max_steps 1
```

验证结果：

- `runtime`：设备检测、内存分配、同步/异步拷贝通过
- `ops`：当前测试集全部通过
- `infer`：Hugging Face 与 LLAISYS token 级严格一致

### 4.4 NVIDIA 路径说明

本次提交遵循“不改坏 CPU + NVIDIA 路径”的约束，因此所有 MetaX 代码都走独立分支。  
但当前机器只有沐曦设备，没有 NVIDIA 硬件，所以本次没有在本机重新跑 `--device nvidia` 的硬件回归。文档只声明：

- NVIDIA 原有实现已被完整阅读并保留
- 构建开关与代码路径没有被 MetaX 改写
- 本机未做新的 NVIDIA 实机验证

## 5. 踩坑记录

### 5.1 MetaX 不是 C++ 层 CUDA 直替

最开始的核心判断问题是：MetaX 到底能不能直接拿 CUDA 代码编过去。实际检查后确认不行：

- `<cuda_runtime.h>` 不能作为稳定适配前提
- `cuBLAS` 也不能直接当成 `mcBLAS`

因此必须走独立后端接入，而不是在 NVIDIA 代码里用宏硬套。

### 5.2 root 用户下 xmake 默认拒绝运行

本机是 root 环境，`xmake` 默认会报错退出，需要显式加：

```bash
XMAKE_ROOT=y
```

这个点如果不写进复现文档，提交后很容易直接卡死在构建阶段。

### 5.3 当前环境没有可直接提交的本地模型目录

为了把推理链路先打通，本次使用公开的小型 Qwen2 测试模型：

- `trl-internal-testing/tiny-Qwen2ForCausalLM-2.5`

这样可以先保证 `test/test_infer.py` 的严格一致性校验可复现；如果老师或助教有本地 Qwen2 模型目录，也可以直接替换 `--model` 参数做同样的验证。

## 6. 已知限制

- `argmax` 与 `self_attention` 在 MetaX 侧当前仍是 host fallback，优先保证链路正确性，不追求这一步的性能最优。
- 当前报告没有新增 NVIDIA 实机回归数据，因为本机没有 NVIDIA 硬件。
- 当前推理验证聚焦 `Qwen2`；不再沿用旧文档里关于 `TinyLlama/Llama` 的完成声明。
- 根目录外部 PDF 保持未跟踪状态，不会随仓库提交。

## 7. 提交物索引

- 提交总览：[`submission_zh.md`](submission_zh.md)
- 复现流程：[`reproduce_zh.md`](reproduce_zh.md)
- PR 文案：[`pr_zh.md`](pr_zh.md)

## 8. 提交边界说明

为保持课程提交 PR 干净，本次仓库提交只保留以下内容：

- 项目 2 第二平台实现代码
- 构建、测试与 Python 桥接相关改动
- 正式提交文档

以下内容不进入仓库提交：

- 本地学习材料
- 复试问答、讲稿与简历草稿
- 外部平台说明 PDF
- handoff 或临时排障文档
