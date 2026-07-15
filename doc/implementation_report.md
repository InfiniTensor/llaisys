# LLAISYS 扩展实现与实验报告

## 1. 报告说明

本报告对应仓库扩展任务的当前实现状态，重点记录以下内容：

- 已完成或部分完成的项目及其对应实现
- 关键功能与当前边界
- 主要实验结果
- 复现方式

本次工作围绕以下方向展开：

- 项目 #1：CPU 优化
- 项目 #2：Nvidia CUDA 后端
- 项目 #3：聊天机器人与采样生成
- 项目 #4：多用户推理服务
- 项目 #5：分布式推理中的张量并行
- 项目 #6：新增模型类型支持入口

## 2. 任务完成情况概览

| 项目 | 当前状态 | 说明 |
| --- | --- | --- |
| #1 CPU 优化 | 已完成 | CPU 主推理路径上的核心算子均已做专门优化 |
| #2 CUDA 集成 | 已完成一类平台 | 已完成 Nvidia 平台；未继续实现其他 CUDA/CUDA-ish 平台 |
| #3 聊天机器人 | 已完成 | 采样、流式输出、CLI、会话管理均已实现 |
| #4 多用户推理服务 | 已完成可运行版本 | 已实现请求池、调度线程、worker 池、会话复用；尚未实现真正的后端 batched decode |
| #5 分布式推理 | 已完成可运行版本 | 已实现 Nvidia/Qwen2 张量并行；当前切分策略对 `tp_size` 有约束 |
| #6 新模型支持 | 已完成基础支持入口 | 已支持 `qwen2`、`llama`、`mistral` 的模型识别与创建入口 |

## 3. 各项目实现情况

### 3.1 项目 #1：Optimize LLAISYS for CPU

#### 3.1.1 算子优化

CPU 路径上已经完成一轮专门优化的算子包括：

- `add`
- `argmax`
- `embedding`
- `linear`
- `rearrange`
- `rms_norm`
- `rope`
- `self_attention`
- `swiglu`
- `sample`

其中，`linear` 是重点优化对象，涉及的主要工作包括：

- OpenMP 并行
- 多级分块
- AVX2/FMA 快路径
- 对部分大规模 `f32 GEMM` 接入 OpenBLAS
- 根据矩阵形状启发式选择内核，而非无条件走 BLAS

相关实现分布在以下路径：

- `src/ops/linear/cpu/linear_cpu.cpp`
- `src/ops/add/cpu/add_cpu.cpp`
- `src/ops/argmax/cpu/argmax_cpu.cpp`
- `src/ops/embedding/cpu/embedding_cpu.cpp`
- `src/ops/rearrange/cpu/rearrange_cpu.cpp`
- `src/ops/rms_norm/cpu/rms_norm_cpu.cpp`
- `src/ops/rope/cpu/rope_cpu.cpp`
- `src/ops/self_attention/cpu/self_attention_cpu.cpp`
- `src/ops/swiglu/cpu/swiglu_cpu.cpp`
- `src/ops/sample/cpu/sample_cpu.cpp`

#### 3.1.2 模型推理路径优化

在 Qwen2 推理路径中，还增加了 buffer 复用机制，以降低频繁创建临时张量带来的额外开销。

已完成的优化包括：

- decode 路径 buffer 复用
- prefill 路径 scratch buffer 复用

相关实现位于：

- `src/models/qwen2/qwen2.cpp`

### 3.2 项目 #2：Integrate CUDA into LLAISYS

#### 3.2.1 Nvidia Runtime API

已完成 Nvidia 平台 Runtime API 的实现，并打通了构建流程。

相关文件包括：

- `xmake/nvidia.lua`
- `xmake.lua`
- `src/device/nvidia/cuda_utils.cuh`
- `src/device/nvidia/nvidia_runtime_api.cu`

同时修复了多设备 Runtime 在错误 device 上创建 CUDA stream 的问题，保证多 GPU 环境下每个 Runtime 使用与自身 device 对应的 stream。

相关实现位于：

- `src/core/runtime/runtime.cpp`

#### 3.2.2 Nvidia 算子实现

CUDA 算子没有集中放在单一 `.cu` 文件中，而是按照算子目录拆分实现。当前已完成的 Nvidia 算子包括：

- `src/ops/add/nvidia`
- `src/ops/argmax/nvidia`
- `src/ops/embedding/nvidia`
- `src/ops/linear/nvidia`
- `src/ops/rearrange/nvidia`
- `src/ops/rms_norm/nvidia`
- `src/ops/rope/nvidia`
- `src/ops/sample/nvidia`
- `src/ops/self_attention/nvidia`
- `src/ops/swiglu/nvidia`
- `src/ops/nvidia/nvidia_common.cuh`

#### 3.2.3 CUDA 推理路径

Qwen2 的 CUDA 推理路径已经打通，能够在 Nvidia 设备上完成真实模型推理与采样生成。

相关路径：

- `src/models/qwen2/qwen2.cpp`
- `python/llaisys/models/qwen2.py`

当前未继续实现第二个 CUDA/CUDA-ish 平台，因此项目 #2 的完成范围限定为 Nvidia。

### 3.3 项目 #3：Build an AI chatbot

#### 3.3.1 随机采样

新增 `sample` 算子，并接入模型生成路径。当前支持：

- `temperature`
- `top-k`
- `top-p`
- 当 `temperature <= 0` 或 `top_k == 1` 时退化为 argmax

相关路径：

- `src/ops/sample/op.cpp`
- `src/ops/sample/cpu/sample_cpu.cpp`
- `src/ops/sample/nvidia/sample_nvidia.cu`
- `python/llaisys/ops.py`

#### 3.3.2 聊天服务与 CLI

已实现 OpenAI 风格的聊天接口：

- `POST /v1/chat/completions`

并支持：

- SSE 流式输出
- 命令行交互式聊天
- 非流式请求

相关路径：

- `python/llaisys/chat_server.py`
- `python/llaisys/chat_cli.py`

#### 3.3.3 会话管理

当前已经实现基础会话管理能力，包括：

- 会话创建、查询、更新、删除
- 会话历史查看
- 重生成上一次助手回复
- 编辑历史用户消息并重新生成
- 基于 `truncate` 的 KV 状态回退

服务端接口包括：

- `GET /v1/sessions`
- `POST /v1/sessions`
- `GET /v1/sessions/{session_id}`
- `PUT /v1/sessions/{session_id}`
- `DELETE /v1/sessions/{session_id}`

CLI 支持：

- `/new`
- `/list`
- `/switch`
- `/history`
- `/regen`
- `/edit`
- `/delete`
- `/session`
- `/help`

### 3.4 项目 #4：Multi-user Inference Service

#### 3.4.1 服务层调度

当前多用户服务采用三层结构：

- HTTP 层：参数校验、建请求对象、返回响应
- 调度层：后台线程从请求池取出请求并分发
- 执行层：worker 持有模型实例并实际执行生成

已实现的能力包括：

- 请求池
- 调度线程
- worker 池
- 微批次出队
- 迭代级连续调度
- 会话与 worker 亲和
- 同一会话前缀复用
- 同一 `session_id` 并发保护

相关路径：

- `python/llaisys/chat_server.py`
- `test/test_chat_api.py`
- `test/chat_test_utils.py`

#### 3.4.2 当前边界

项目 #4 当前已经具备“服务层 continuous batching”的核心形态，但尚未实现以下能力：

- 单个后端模型对象内的真正 batched decode
- batched KV-cache
- 多请求单轮合并后的统一 batch forward
- batched matmul 路径

也就是说，当前调度器会以迭代级粒度推进请求，但单个 worker 一次仍然只执行一个请求的一步生成。

### 3.5 项目 #5：Distributed Inference

#### 3.5.1 张量并行实现

当前已经实现 Nvidia 平台上的 Qwen2 张量并行版本，使用 `torch.distributed` 和 NCCL。

模型入口位于：

- `python/llaisys/models/tensor_parallel.py`
- `python/llaisys/models/__init__.py`

支持的能力包括：

- `reset`
- `truncate`
- `generate_next`
- `generate`
- `stream_generate`

服务端已支持以下参数：

- `--tp-size`
- `--tp-device-ids`

#### 3.5.2 当前切分策略与限制

当前实现采用均匀切分策略：

- `Q heads` 均匀切分
- `KV heads` 均匀切分
- `MLP intermediate` 均匀切分

因此，`tp_size` 需要同时满足这些相关维度的均匀切分要求。对当前使用的 `DeepSeek-R1-Distill-Qwen-1.5B` 而言：

- `num_attention_heads = 12`
- `num_key_value_heads = 2`
- `intermediate_size = 8960`

在当前实现下，允许的 `tp_size` 为：

- `1`
- `2`

因此，两卡张量并行可以正常工作，四卡在当前实现下不支持。

#### 3.5.3 当前边界

当前项目 #5 尚不包括：

- MPI/CPU 分布式推理
- 更复杂的 KV 复制式张量并行
- 针对低 `num_key_value_heads` 模型的更高卡数切分策略

### 3.6 项目 #6：Support New Models

当前已完成统一模型创建入口，并支持自动识别 `config.json` 中的 `model_type`。

目前支持的模型类型包括：

- `qwen2`
- `llama`
- `mistral`

相关路径：

- `python/llaisys/models/__init__.py`

该部分工作的重点是先统一接入路径与创建流程。当前真实完整验证仍以 Qwen2 为主。

## 4. 关键实验结果

### 4.1 CPU `linear` 初始基线

在最早的 CPU `linear` profile 中，观测到如下基线：

- 矩阵形状：`(64, 512) x (512, 512)`
- PyTorch：约 `0.269 ms`
- LLAISYS：约 `25.427 ms`

### 4.2 CPU `linear` 优化后结果

在 CPU 优化后，小规模 `linear` 的结果达到过：

- 矩阵形状：`(64, 512) x (512, 512)`
- LLAISYS：约 `0.0568 ms`

### 4.3 接近真实模型形状的 `linear`

对更接近真实模型的矩阵形状，测得：

- `out=(198, 8960), x=(198, 1536), w=(8960, 1536)`
  - Torch：`4.88102 ms`
  - LLAISYS：`10.06528 ms`

- `out=(198, 1536), x=(198, 8960), w=(1536, 8960)`
  - Torch：`1.96279 ms`
  - LLAISYS：`4.23397 ms`

### 4.4 CPU 真实模型推理

在 `DeepSeek-R1-Distill-Qwen-1.5B` 上，短生成曾测得：

- 生成 8 个 token
  - `short_generate_s=0.600684`
  - `short_tok_per_s=13.318142`

官方回归测试的一次结果：

- Hugging Face：约 `0.71 s`
- LLAISYS CPU：约 `0.72 s`
- token 序列：完全一致

### 4.5 CUDA 真实模型推理

单卡 Nvidia 回归测试的一次结果：

- Hugging Face：约 `0.79 s`
- LLAISYS Nvidia：约 `0.14 s`
- token 序列：完全一致

### 4.6 两卡张量并行

两卡张量并行 smoke test 已通过，标准 prompt 的生成 token 与期望结果一致。

## 5. 复现方式

### 5.1 构建

启用 CPU BLAS 与 Nvidia：

```bash
xmake f --cpu-blas=y --openblas-prefix="$CONDA_PREFIX" --nv-gpu=y -cv
xmake
xmake install
```

若不启用 BLAS：

```bash
xmake f --nv-gpu=y -cv
xmake
xmake install
```

### 5.2 CPU 回归测试

```bash
conda run -n llaisys env PYTHONPATH=python:test python test/test_runtime.py --device cpu
conda run -n llaisys env PYTHONPATH=python:test python test/ops/linear.py --device cpu
conda run -n llaisys env PYTHONPATH=python:test python test/ops/sample.py --device cpu
conda run -n llaisys env PYTHONPATH=python:test python test/test_generate_sampling.py --device cpu --model [模型目录]
conda run -n llaisys env PYTHONPATH=python:test python test/test_infer.py --device cpu --model [模型目录] --test --max_steps 8
```

### 5.3 Nvidia 回归测试

```bash
conda run -n llaisys env PYTHONPATH=python:test python test/test_runtime.py --device nvidia
conda run -n llaisys env PYTHONPATH=python:test python test/ops/add.py --device nvidia
conda run -n llaisys env PYTHONPATH=python:test python test/ops/argmax.py --device nvidia
conda run -n llaisys env PYTHONPATH=python:test python test/ops/embedding.py --device nvidia
conda run -n llaisys env PYTHONPATH=python:test python test/ops/linear.py --device nvidia
conda run -n llaisys env PYTHONPATH=python:test python test/ops/rms_norm.py --device nvidia
conda run -n llaisys env PYTHONPATH=python:test python test/ops/rope.py --device nvidia
conda run -n llaisys env PYTHONPATH=python:test python test/ops/swiglu.py --device nvidia
conda run -n llaisys env PYTHONPATH=python:test python test/ops/self_attention.py --device nvidia
conda run -n llaisys env PYTHONPATH=python:test python test/test_generate_sampling.py --device nvidia --model [模型目录]
conda run -n llaisys env PYTHONPATH=python:test python test/test_infer.py --device nvidia --model [模型目录] --test --max_steps 8
```

### 5.4 聊天与会话管理测试

```bash
conda run -n llaisys env PYTHONPATH=python:test python test/test_chat_api.py
conda run -n llaisys env PYTHONPATH=python:test python test/test_chat_cli.py
```

### 5.5 张量并行测试

```bash
conda run -n llaisys env PYTHONPATH=python:test python test/test_tensor_parallel.py --model [模型目录] --tp-size 2 --max-steps 8
```

### 5.6 启动单卡 CUDA 聊天服务

```bash
PYTHONPATH=python python -m llaisys.chat_server \
  --model-path [模型目录] \
  --device nvidia \
  --device-id 0 \
  --host 127.0.0.1 \
  --port 8000 \
  --num-workers 1 \
  --max-batch-size 1 \
  --batch-wait-ms 0
```

### 5.7 启动两卡张量并行聊天服务

```bash
PYTHONPATH=python python -m llaisys.chat_server \
  --model-path [模型目录] \
  --device nvidia \
  --device-id 0 \
  --tp-size 2 \
  --tp-device-ids 0,1 \
  --host 127.0.0.1 \
  --port 8000 \
  --num-workers 1 \
  --max-batch-size 2 \
  --batch-wait-ms 5
```

### 5.8 交互式 CLI

```bash
PYTHONPATH=python python -m llaisys.chat_cli \
  --url http://127.0.0.1:8000 \
  --model llaisys-qwen2 \
  --max-tokens 512
```

### 5.9 curl 示例

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "llaisys-qwen2",
    "session_id": "demo",
    "messages": [{"role": "user", "content": "请解释操作系统的页表机制"}],
    "max_tokens": 512,
    "temperature": 0.8,
    "top_p": 0.8,
    "top_k": 50,
    "stream": true
  }'
```

服务统计接口：

```bash
curl http://127.0.0.1:8000/v1/service/stats
```

## 6. 当前边界说明

当前仓库已经从基础的 CPU/Qwen2 推理项目扩展为：

- CPU 优化版本
- Nvidia CUDA 版本
- 支持采样、流式输出和会话管理的聊天服务
- 支持多用户请求池与服务层连续调度的推理服务
- 支持两卡张量并行的 Nvidia 推理版本
- 支持多模型类型统一创建入口的版本

尚未纳入当前完成范围的内容主要包括：

- 第二个 CUDA/CUDA-ish 平台
- 真正后端层面的 batched decode / batched KV-cache
- 更复杂的 KV 复制式张量并行
- MPI/CPU 分布式推理
