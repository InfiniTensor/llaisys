# 项目二与项目三完成报告

## 一、完成概要

本次完成了 README 中的项目二和项目三，主要成果如下：

1. 在 LLAISYS 中完成了双 GPU 后端接入，支持 `NVIDIA` 与 `MetaX` 两个平台。
2. 完成了 `Qwen2` 模型在 LLAISYS 后端的推理实现，支持权重加载、KV-Cache 和逐 token 解码。
3. 完成了项目二要求的核心算子实现与接入，包括 `add`、`argmax`、`embedding`、`linear`、`rms_norm`、`rope`、`self_attention`、`swiglu`。
4. 完成了项目三要求的随机采样功能，支持 `temperature`、`top-k`、`top-p`。
5. 完成了聊天服务与交互界面，提供 `FastAPI` 服务端、命令行客户端和 Web 界面，并支持流式输出。
6. 编写了统一的推理 benchmark 脚本，用于比较 `Torch` 与 `LLAISYS` 的输出对齐情况和吞吐表现。

当前工程已经能够在本地 `NVIDIA` 平台和远程 `MetaX` 平台完成端到端模型推理，并具备聊天服务的基本交付能力。

## 二、开发环境

### 1. 本地开发与验证环境

- 操作系统：Linux
- GPU：NVIDIA RTX 4060
- CUDA：本地安装 CUDA 工具链
- 构建工具：`xmake`
- Python：Python 3.x
- 主要依赖：`transformers`、`huggingface_hub`、`fastapi`、`uvicorn`

### 2. 远程 MetaX 验证环境

- 操作系统：Linux
- GPU：MetaX GPU
- 开发环境：`MACA / mcPyTorch`
- 头文件与库路径：远程环境已安装对应 MetaX SDK

### 3. 模型与测试对象

- 模型：`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- 权重格式：`safetensors`
- 主要数据类型：`bf16`

## 三、项目二具体实现

### 1. 双平台 Runtime 与构建链路

在 LLAISYS 原有 CPU 框架基础上，补充了 `NVIDIA` 与 `MetaX` 两套设备后端：

- 实现了 `nvidia` Runtime API 与 `metax` Runtime API。
- 在构建系统中增加了平台开关，支持通过 `xmake` 分别编译 `NVIDIA` 与 `MetaX` 后端。
- 在 Python 侧补充设备映射，使测试脚本和推理脚本能够通过 `--device nvidia` 与 `--device metax` 调用对应后端。

### 2. 核心算子实现

项目二要求的核心算子已经在 GPU 后端完成实现，并接入统一算子分发路径。主要包括：

- `add`
- `argmax`
- `embedding`
- `linear`
- `rms_norm`
- `rope`
- `self_attention`
- `swiglu`

其中：

- `NVIDIA` 路径主要采用 CUDA 风格实现，并在 `linear` 等算子中使用官方库加速。
- `MetaX` 路径尽量对齐 CUDA 实现风格，优先使用 MetaX 官方 API 与 `mcBLAS`。
- 针对 MetaX 平台 `warp=64` 的特性，对部分 kernel 的 block 配置和规约方式做了适配。

### 3. 模型推理实现

围绕 `Qwen2` 模型，完成了 LLAISYS 后端推理链路：

- 在 C/C++ 后端实现模型结构、张量组织和推理逻辑。
- 实现 `safetensors` 权重加载接口。
- 实现 KV-Cache，支持逐 token 解码。
- 在 Python 包装层中完成 `Qwen2` 模型封装，支持 `generate` 与 `generate_stream`。

### 4. 功能验证情况

项目二完成后，已完成以下验证：

- Runtime 测试：验证设备运行时接口可用。
- 算子测试：各核心算子均有对应测试脚本，可在指定设备上运行。
- 推理测试：`test/test_infer.py` 可用于验证 LLAISYS 输出是否与 Torch 对齐。
- Benchmark 测试：`test/benchmark_infer.py` 用于比较 Torch 与 LLAISYS 的推理性能与吞吐，输出对齐由 `test/test_infer.py` 单独负责验证。

本地 `NVIDIA` 平台最新 benchmark 结果如下：

| Case | Torch mean(ms) | Torch tok/s | LLAISYS mean(ms) | LLAISYS tok/s | speedup |
|---|---:|---:|---:|---:|---:|
| short/32 | 810.54 | 39.48 | 495.97 | 64.52 | 1.63x |
| short/64 | 1563.33 | 40.94 | 1007.77 | 63.51 | 1.55x |
| short/128 | 2079.48 | 38.95 | 1280.56 | 63.25 | 1.62x |
| medium/32 | 786.33 | 40.70 | 506.45 | 63.19 | 1.55x |
| medium/64 | 1802.99 | 35.50 | 1029.44 | 62.17 | 1.75x |
| medium/128 | 3219.73 | 39.75 | 2114.44 | 60.54 | 1.52x |
| long/32 | 1032.12 | 31.00 | 522.34 | 61.26 | 1.98x |
| long/64 | 1616.44 | 39.59 | 1040.72 | 61.50 | 1.55x |
| long/128 | 3160.70 | 40.50 | 2155.55 | 59.38 | 1.47x |

吞吐汇总如下：

- Torch total throughput：`38.89 tok/s`
- LLAISYS total throughput：`61.56 tok/s`
- Overall speedup：`1.58x`

从这组结果可以看到，LLAISYS 在本地 `NVIDIA` 平台上已经取得了稳定的端到端推理性能优势。

远程 `MetaX` 平台最新 benchmark 结果如下：

| Case | Torch mean(ms) | Torch tok/s | LLAISYS mean(ms) | LLAISYS tok/s | speedup |
|---|---:|---:|---:|---:|---:|
| short/32 | 864.34 | 37.02 | 356.17 | 89.85 | 2.43x |
| short/64 | 1749.20 | 36.59 | 818.50 | 78.19 | 2.14x |
| short/128 | 2173.61 | 37.27 | 1105.36 | 73.28 | 1.97x |
| medium/32 | 865.01 | 36.99 | 437.44 | 73.15 | 1.98x |
| medium/64 | 1721.78 | 37.17 | 977.52 | 65.47 | 1.76x |
| medium/128 | 3439.50 | 37.21 | 2386.28 | 53.64 | 1.44x |
| long/32 | 863.88 | 37.04 | 516.00 | 62.02 | 1.67x |
| long/64 | 1724.36 | 37.12 | 1129.42 | 56.67 | 1.53x |
| long/128 | 3424.45 | 37.38 | 2703.57 | 47.34 | 1.27x |

吞吐汇总如下：

- Torch total throughput：`37.14 tok/s`
- LLAISYS total throughput：`59.92 tok/s`
- Overall speedup：`1.61x`

从这组结果可以看到，LLAISYS 在远程 `MetaX` 平台上同样取得了稳定的端到端推理性能优势。结合 `test/test_infer.py` 的对齐测试，可以说明项目二的双平台推理链路已经打通并完成验证。

## 四、项目三具体实现

### 1. 随机采样

在模型推理接口中补充了随机采样逻辑，支持以下参数：

- `temperature`
- `top-k`
- `top-p`

当参数配置为 `top_k=1, top_p=1.0, temperature=1.0` 时，系统工作在确定性贪心解码模式，可用于和 Torch 做严格 token 对齐测试；其他配置可用于更自然的聊天生成。

### 2. 聊天服务端

实现了基于 `FastAPI` 的聊天服务端，主要能力包括：

- 提供 `/v1/chat/completions` 接口
- 接口风格对齐 OpenAI Chat Completion
- 支持普通返回模式
- 支持基于 `text/event-stream` 的流式输出
- 支持通过请求参数控制 `top-k`、`top-p`、`temperature`、`max_tokens`

服务端入口文件为：

- `test/chat_server.py`

### 3. 命令行交互

实现了命令行聊天客户端，支持：

- 向服务端发送多轮消息
- 保持对话历史
- 支持普通模式和流式模式
- 支持 `/reset` 清空历史、`/exit` 退出

对应文件为：

- `test/chat_cli.py`

### 4. Web 交互界面

实现了简单的 Web 聊天页面，支持：

- 输入对话消息
- 设置 `top-k`、`top-p`、`temperature`
- 切换是否流式输出
- 与 `FastAPI` 服务端联动完成对话

对应文件为：

- `test/chat_web.html`

### 5. 项目三完成情况

目前，项目三已经完成“可采样、可服务、可交互”的基础目标：

- 模型可以通过 LLAISYS 后端执行聊天生成。
- 服务端可以接收 HTTP 请求并返回响应。
- 命令行和 Web 端都可以与服务端交互。
- 系统支持单用户场景下的连续对话与流式输出。

## 五、复现流程

### 1. NVIDIA 平台构建与测试

```bash
cd ~/llaisys
xmake f -c -m release --nv-gpu=y --mx-gpu=n
xmake -r && xmake install
```

运行推理对齐测试：

```bash
python test/test_infer.py \
  --device nvidia \
  --model ~/model_pkg/DeepSeek-R1-Distill-Qwen-1.5B/ \
  --test
```

运行 Torch 与 LLAISYS 的推理 benchmark：

```bash
python test/benchmark_infer.py \
  --device nvidia \
  --model ~/model_pkg/DeepSeek-R1-Distill-Qwen-1.5B/
```

### 2. MetaX 平台构建与测试

在远程 MetaX 服务器上执行：

```bash
cd ~/llaisys
xmake f -c -m release --mx-gpu=y --nv-gpu=n
xmake -r && xmake install
```

运行推理对齐测试：

```bash
python test/test_infer.py \
  --device metax \
  --model ~/model_pkg/DeepSeek-R1-Distill-Qwen-1.5B/ \
  --test
```

运行 benchmark：

```bash
python test/benchmark_infer.py \
  --device metax \
  --model ~/model_pkg/DeepSeek-R1-Distill-Qwen-1.5B/
```

### 3. 聊天服务复现

启动服务端：

```bash
python test/chat_server.py \
  --device nvidia \
  --model ~/model_pkg/DeepSeek-R1-Distill-Qwen-1.5B/
```

命令行客户端连接服务端：

```bash
python test/chat_cli.py --stream
```

Web 端使用方法：

1. 启动 `chat_server.py`
2. 浏览器访问 `http://127.0.0.1:8000/`
3. 在页面中输入消息并发起对话

## 结论

项目二已经完成 LLAISYS 在 `NVIDIA` 与 `MetaX` 双 GPU 平台上的推理后端集成，完成了核心算子、运行时接口和模型推理链路的实现与验证。项目三在此基础上完成了随机采样、聊天服务端、CLI 与 Web UI 的实现，使系统具备了单用户对话式推理的基础能力。

当前代码已经具备提交条件，并能够作为后续性能优化和工程化完善的基础版本。
