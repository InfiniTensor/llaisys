# LLAISYS 项目报告

## 环境信息

- **OS**: WSL2 Ubuntu (Linux 6.6)
- **GPU**: NVIDIA GeForce RTX 3050 (4GB 显存)
- **CUDA**: CUDA Toolkit 12.x, Driver 591.86
- **CPU**: x86_64, 支持 AVX2/FMA
- **构建系统**: xmake
- **模型**: DeepSeek-R1-Distill-Qwen-1.5B (BF16, 28层, hidden_size=1536)

---

## 项目 #1：CPU 推理优化

### 完成功能

1. **OpenMP 多线程并行**
   - 为 `linear`、`embedding`、`rms_norm`、`rope`、`self_attention`、`swiglu` 等算子添加了 OpenMP 并行化
   - 矩阵乘法的外层循环使用 `#pragma omp parallel for` 分配到多核执行

2. **AVX2/FMA SIMD 向量化**
   - `linear` 算子的内积计算使用 AVX2 256-bit 向量指令，每次处理 8 个 float
   - 使用 FMA（Fused Multiply-Add）指令 `_mm256_fmadd_ps` 减少指令数
   - BF16 数据类型支持 SIMD 批量转换

3. **OpenBLAS 集成**
   - `linear` 算子在 FP32 模式下调用 `cblas_sgemm`，利用高度优化的 BLAS 库
   - BF16/FP16 数据先转换为 FP32，再调用 OpenBLAS 计算

### 优化效果

CPU 推理速度相比朴素实现有显著提升，`linear` 算子（占推理总时间 ~80%）获得最大加速。

### 使用方法

```bash
# 构建（默认启用 CPU 优化）
xmake f -c
xmake
xmake install
pip install ./python/

# 运行推理测试
python test/test_infer.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --test --device cpu
```

---

## 项目 #2：CUDA 集成与 GPU 推理加速

### 完成功能

1. **xmake CUDA 构建配置** (`xmake/nvidia.lua`)
   - 配置 CUDA 编译规则，支持 `.cu` 文件编译
   - 自动链接 `cudart` 和 `cublas` 库
   - 通过 `--nv-gpu=y` 编译选项开关 CUDA 支持
   - 自动生成 `build_config.h`，定义 `ENABLE_NVIDIA_API` 宏

2. **CUDA Runtime API** (`src/device/nvidia/nvidia_runtime_api.cu`)
   - 实现了完整的设备管理 API：`getDeviceCount`、`setDevice`、`createStream`、`destroyStream`
   - 实现了内存管理 API：`mallocDevice`、`freeDevice`、`mallocHost`、`freeHost`
   - 实现了数据传输 API：`memcpySync`、`memcpyAsync`（支持 H2D、D2H、D2D）
   - `Context::setDevice` 支持延迟初始化，在运行时动态探测 GPU 设备

3. **10 个 CUDA 算子实现**

   | 算子 | 实现文件 | 关键技术 |
   |------|----------|----------|
   | add | `src/ops/add/cuda/add_cuda.cu` | 逐元素并行 kernel |
   | embedding | `src/ops/embedding/cuda/embedding_cuda.cu` | 按行并行查表 |
   | linear | `src/ops/linear/cuda/linear_cuda.cu` | **cuBLAS cublasGemmEx**，BF16/FP16 直接使用 Tensor Core |
   | rms_norm | `src/ops/rms_norm/cuda/rms_norm_cuda.cu` | 共享内存归约求平方和 |
   | rope | `src/ops/rope/cuda/rope_cuda.cu` | 按 (position, head, dim) 三维并行 |
   | self_attention | `src/ops/self_attention/cuda/self_attention_cuda.cu` | 共享内存 Q 缓存 + warp 级 shuffle 归约 softmax |
   | swiglu | `src/ops/swiglu/cuda/swiglu_cuda.cu` | 逐元素并行 SiLU×gate |
   | argmax | `src/ops/argmax/cuda/argmax_cuda.cu` | 并行归约求最大值 |
   | rearrange | `src/ops/rearrange/cuda/rearrange_cuda.cu` | 按线性索引映射多维步长 |
   | sample | `src/ops/sample/cuda/sample_cuda.cu` | GPU 端 Temperature/Top-K/Top-P 采样 |

4. **性能优化**
   - **BF16 原生 Tensor Core 加速**：`cublasGemmEx` 直接接受 BF16 输入，利用 RTX 3050 (SM 86) 的 Ampere Tensor Core，无需 FP32 中转
   - **工作空间预分配**：模型 forward 中的中间张量预先分配并复用，消除每个 token ~196 次 `cudaMalloc/cudaFree`
   - **异步 D2D 拷贝**：KV Cache 写入使用 `cudaMemcpyAsync`，避免不必要的 CPU-GPU 同步
   - **消除冗余 memcpy**：attention 输出直接传给 linear 算子，跳过不必要的 D2D 拷贝

5. **Qwen2 模型 CUDA 推理** (`src/models/qwen2.cpp`)
   - 完整的 28 层 Transformer 前向传播在 GPU 上执行
   - KV Cache 存储在 GPU 显存中，支持自回归生成
   - 支持 argmax 和随机采样两种生成模式

### 性能结果

| 方案 | 生成 90 tokens 耗时 | tokens/sec |
|------|---------------------|------------|
| HuggingFace PyTorch (参考) | ~4.7s | ~19 |
| **LLAISYS GPU** | **~5.4s** | **~17** |

LLAISYS GPU 推理速度接近 HuggingFace PyTorch，仅慢约 16%。

### 使用方法

```bash
# 构建（启用 CUDA）
xmake f --nv-gpu=y -c
xmake
xmake install
pip install ./python/

# 运行 CUDA Runtime 测试
python test/test_runtime.py --device nvidia

# 运行算子测试
python test/test_ops.py --device nvidia

# 运行推理正确性测试
python test/test_infer.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --test --device nvidia
```

---

## 项目 #3：AI 聊天机器人

### 完成功能

1. **随机采样算子** (`src/ops/sample/`)
   - **Temperature 采样**：通过温度参数控制生成随机性，logits 除以 temperature 后进行 softmax
   - **Top-K 采样**：只保留概率最高的 K 个 token，其余置零后重新归一化
   - **Top-P (Nucleus) 采样**：按概率从高到低累加，保留累积概率达到 P 的最小 token 集合
   - 同时提供 CPU 和 CUDA 两个版本

2. **FastAPI 聊天服务器** (`python/llaisys/server.py`)
   - **OpenAI 兼容 API**：实现 `/v1/chat/completions` 端点，兼容 OpenAI Chat Completion 格式
   - **流式输出 (SSE)**：支持 `stream: true`，通过 Server-Sent Events 实时逐 token 推送回复
   - **非流式输出**：支持 `stream: false`，一次返回完整回复
   - **模型列表接口**：`/v1/models` 返回可用模型
   - **GPU 支持**：`--device nvidia` 参数启用 GPU 加速推理
   - **线程安全**：全局互斥锁确保模型推理的线程安全

3. **Web 聊天界面** (`python/llaisys/static/index.html`)
   - 现代化单页 Web UI，支持发送消息和接收回复
   - **流式打字效果**：回复逐字显示，类似 ChatGPT 体验
   - **对话历史**：前端维护完整 messages 数组，支持多轮对话上下文
   - **参数调节**：可调整 Temperature、Top-K、Top-P、Max Tokens
   - **清空对话**：一键清除对话历史

### 架构设计

```
┌──────────────┐     HTTP/SSE      ┌──────────────────┐     C API      ┌─────────────┐
│  Web UI      │ ◄──────────────►  │  FastAPI Server   │ ◄────────────► │  LLAISYS    │
│  (HTML/JS)   │   /v1/chat/       │  (Python)         │   ctypes       │  C++ Backend│
│              │   completions     │                   │                │  (CPU/CUDA) │
└──────────────┘                   └──────────────────┘                └─────────────┘
```

### 使用方法

```bash
# 构建并安装
xmake f --nv-gpu=y -c
xmake
xmake install
pip install ./python/

# 启动聊天服务器（GPU 模式）
python -m llaisys.server --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --device nvidia --port 8000

# 启动聊天服务器（CPU 模式）
python -m llaisys.server --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --device cpu --port 8000
```

启动后打开浏览器访问 `http://localhost:8000` 即可使用聊天界面。

也可通过 curl 直接调用 API：

```bash
# 非流式请求
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"你好"}],"max_tokens":100,"stream":false}'

# 流式请求
curl -N -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"你好"}],"max_tokens":100,"stream":true}'
```

---

## 文件清单

### 项目 #1 新增/修改文件

- `src/ops/add/cpu/add_cpu.cpp` — CPU add 算子（OpenMP）
- `src/ops/linear/cpu/linear_cpu.cpp` — CPU linear 算子（OpenBLAS + AVX2/FMA）
- `src/ops/rms_norm/cpu/rms_norm_cpu.cpp` — CPU rms_norm 算子
- `src/ops/rope/cpu/rope_cpu.cpp` — CPU rope 算子
- `src/ops/self_attention/cpu/self_attention_cpu.cpp` — CPU self_attention 算子
- `src/ops/swiglu/cpu/swiglu_cpu.cpp` — CPU swiglu 算子
- `src/ops/embedding/cpu/embedding_cpu.cpp` — CPU embedding 算子
- `src/ops/argmax/cpu/argmax_cpu.cpp` — CPU argmax 算子
- `src/ops/rearrange/cpu/rearrange_cpu.cpp` — CPU rearrange 算子
- `xmake/cpu.lua` — CPU 编译配置

### 项目 #2 新增文件

- `xmake/nvidia.lua` — CUDA 编译配置
- `src/device/nvidia/nvidia_runtime_api.cu` — CUDA Runtime API 实现
- `src/ops/add/cuda/add_cuda.cu` — CUDA add 算子
- `src/ops/embedding/cuda/embedding_cuda.cu` — CUDA embedding 算子
- `src/ops/linear/cuda/linear_cuda.cu` — CUDA linear 算子（cuBLAS Tensor Core）
- `src/ops/rms_norm/cuda/rms_norm_cuda.cu` — CUDA rms_norm 算子
- `src/ops/rope/cuda/rope_cuda.cu` — CUDA rope 算子
- `src/ops/self_attention/cuda/self_attention_cuda.cu` — CUDA self_attention 算子
- `src/ops/swiglu/cuda/swiglu_cuda.cu` — CUDA swiglu 算子
- `src/ops/argmax/cuda/argmax_cuda.cu` — CUDA argmax 算子
- `src/ops/rearrange/cuda/rearrange_cuda.cu` — CUDA rearrange 算子
- `src/ops/sample/cuda/sample_cuda.cu` — CUDA sample 算子
- `src/models/qwen2.hpp` — Qwen2 模型头文件（含工作空间预分配）
- `src/models/qwen2.cpp` — Qwen2 模型实现（GPU forward）
- `src/core/context/context.cpp` — Context 延迟初始化修复

### 项目 #3 新增文件

- `src/ops/sample/cpu/sample_cpu.cpp` — CPU sample 算子（Temperature/Top-K/Top-P）
- `src/ops/sample/cuda/sample_cuda.cu` — CUDA sample 算子
- `src/ops/sample/op.cpp` — sample 算子调度
- `python/llaisys/server.py` — FastAPI 聊天服务器
- `python/llaisys/static/index.html` — Web 聊天界面
- `python/llaisys/libllaisys/qwen2.py` — Qwen2 ctypes 绑定
- `src/llaisys/qwen2.cc` — Qwen2 C API 实现
