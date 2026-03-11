# MetaX/MACA 第二平台实现说明

## 1. 目标

本说明对应当前仓库已经落地的第二平台实现，而不是停留在“只交设计稿”的阶段。目标很明确：

- 不改坏已有 `CPU + NVIDIA` 路径
- 在 LLAISYS 现有抽象上新增独立 `METAX` 设备类型
- 先把 `runtime -> ops -> infer` 主链路打通
- 优先保证可测试、可提交、可复现

## 2. 为什么不复用 NVIDIA 后端

平台检查后的结论是：MetaX 不能直接当成 C++ 层的 CUDA 替身。

主要原因：

- SDK 头文件是 `<mcr/mc_runtime.h>` 与 `<mcblas/mcblas.h>`
- 编译器是 `mxcc`，不是 `nvcc`
- 运行库在 `/opt/maca/lib` 与 `/opt/mxdriver/lib`
- 设备管理命令是 `mx-smi`，不是 `nvidia-smi`

因此，本次实现必须新增独立 `metax` 后端，而不是在 `nvidia` 代码里做一堆条件编译硬适配。

## 3. 设备与构建设计

### 3.1 设备枚举

在 `llaisysDeviceType_t` 中新增：

- `LLAISYS_DEVICE_METAX`

保留已有 `CPU` 与 `NVIDIA` 的枚举值和分发逻辑不变。

### 3.2 构建开关

在 `xmake.lua` 中新增：

- `--metax-gpu=y`

同时新增 `xmake/metax.lua`：

- 用 `mxcc` 编译 MetaX `.cu`
- 注入 `/opt/maca/include`
- 链接 `/opt/maca/lib`、`/opt/mxdriver/lib`
- 显式链接 `mcblas` 与 `mcruntime`

另外增加了一个最小的 MetaX link stub，避免设备对象在共享库最终链接阶段被漏掉。

## 4. Runtime 适配

MetaX runtime 与 LLAISYS 现有抽象一一对应，映射如下：

| LLAISYS | MetaX/MACA |
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

实现时还补了两个关键约束：

- 错误处理统一经过 `mcGetErrorString` / `mcblasGetStatusString`
- `mcblasCreate` 之前先显式激活设备，避免 handle 落到错误设备上

## 5. 资源管理设计

MetaX 资源层放在 `src/device/metax/`，主要职责是：

- 维护 `thread_local` 的 `mcblasHandle`
- 按设备号缓存 handle，避免重复创建
- 每次算子调用前把当前 stream 绑定到 handle

这样可以和现有 Context / Runtime 模型对齐，不需要让上层模型逻辑感知平台差异。

## 6. 算子实现策略

MetaX 算子统一放在 `src/ops/metax/ops_metax.cu`，由各算子的 `op.cpp` 做设备分发。

当前支持矩阵如下：

| 算子 | 实现方式 | 备注 |
| --- | --- | --- |
| `add` | MetaX kernel | 设备端逐元素计算 |
| `embedding` | MetaX kernel | 设备端 gather |
| `linear` | `mcblasGemmEx` | bias 追加一个轻量 kernel |
| `rms_norm` | MetaX kernel | 行级归一化 |
| `rope` | MetaX kernel | 设备端旋转位置编码 |
| `swiglu` | MetaX kernel | 设备端激活与逐元素乘 |
| `argmax` | host fallback | 优先保证正确性 |
| `self_attention` | host fallback | 优先保证主链路可测 |

### 6.1 `linear` 的 dtype 和数值策略

`linear` 是第二平台最核心的算子，当前策略是：

- dtype 映射到 `MACA_R_32F` / `MACA_R_16F` / `MACA_R_16BF`
- 半精度与 bf16 计算类型使用 `MCBLAS_COMPUTE_32F`
- `float32` 路径使用 `MCBLAS_COMPUTE_32F_PEDANTIC`
- 数学模式固定为 `MCBLAS_PEDANTIC_MATH`

这么做不是为了追求极限性能，而是为了先压住和 PyTorch 对照时的微小数值漂移，保证课程测试更稳定。

### 6.2 为什么允许 host fallback

项目当前阶段的目标是让第二平台达到“可实际测试”状态，而不是一次性把所有算子都做成高性能设备实现。  
因此：

- `argmax` 先用 host fallback 打通最终 token 读取
- `self_attention` 先用 host fallback 保证推理链路正确

后续如果继续做性能优化，再把这两部分替换成真正的 MetaX kernel 即可。

## 7. Python 与测试桥接

Python 层同步新增：

- `DeviceType.METAX`
- `--device metax`

测试桥接里有一个容易误解的点：

- `llaisys` 自己的设备枚举是 `METAX`
- Hugging Face / PyTorch 对照仍然走 `torch.cuda`

原因是本机 MetaX PyTorch 暴露的是 CUDA 语义设备接口，而不是新的 `torch.device("metax")`。

## 8. 当前验证结果

本机已通过：

```bash
XMAKE_ROOT=y xmake f --metax-gpu=y -cv
XMAKE_ROOT=y xmake -r
XMAKE_ROOT=y xmake install

python test/test_runtime.py --device metax
python test/test_ops.py --device metax
python test/test_infer.py --device metax --test --model_id trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 --prompt hi --max_steps 1
```

同时补做了 CPU 基线验证：

```bash
python test/test_tensor.py
python test/test_runtime.py --device cpu
python test/test_ops.py --device cpu
python test/test_infer.py --device cpu --test --model_id trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 --prompt hi --max_steps 1
```

## 9. 当前限制

- `argmax` 与 `self_attention` 仍是 host fallback
- 当前机器没有 NVIDIA 硬件，所以没有新做 NVIDIA 实机回归
- 当前推理链路实际验证的是 `Qwen2`
- 根目录外部 PDF 不纳入仓库提交
