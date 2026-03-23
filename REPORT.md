# LLAISYS 项目报告

## 一、已完成工作总览

| 模块 | 说明 |
|------|------|
| 作业 #0-#3（基础） | 张量、算子、模型推理全部完成 |
| 项目 #2：多平台 CUDA 适配 | Nvidia + 天数 Iluvatar CoreX 双平台 |
| 项目 #3：AI 聊天机器人 | 服务器 + 前端 + 流式输出 + 会话管理 + KV 复用 |
| 项目 #4：多用户推理服务 | 调度器 + 连续批处理 + 共享模型池 + KV 感知路由 |
| 项目 #5：分布式推理 | 通信层 + NCCL 后端 + 张量并行 |

## 二、作业阶段

**作业 #1：张量** — 实现了张量的核心操作：`load`、`isContiguous`、`view`、`permute`、`slice`。所有测试通过。

**作业 #2：算子** — 实现了 9 个 CPU 算子：`add`、`argmax`、`embedding`、`linear`、`rearrange`、`rms_norm`、`rope`、`self_attention`、`swiglu`。支持 Float32/Float16/BFloat16 数据类型，全部测试通过。

**作业 #3：大语言模型推理** — 实现了 DeepSeek-R1-Distill-Qwen-1.5B 模型的完整推理链路：C++ Decoder 实现（Transformer 前向传播 + KV Cache）、C API 导出 + Python ctypes 封装、端到端推理输出与 PyTorch 完全一致。

## 三、项目阶段

**项目 #2：多平台 CUDA 适配**

在 Nvidia GPU 和天数 Iluvatar CoreX GPU 两个平台上实现 CUDA 加速推理。

实现方案：
- Nvidia 平台：实现 CUDA Runtime API + 9 个 CUDA 算子内核，使用 nvcc 编译
- 天数 Iluvatar CoreX 平台：采用 kernel 零复制策略，直接复用 `nvidia::` 命名空间的 CUDA 内核，使用 `clang++ -x cuda --cuda-gpu-arch=ivcore10` 编译

关键问题与解决：

| 问题 | 解决方案 |
|------|----------|
| xmake 自动调用 nvcc 而非 clang++ | 使用 `on_build()` 手动控制编译 |
| xmake 注入 `-lcudadevrt` | 不注册 .cu 文件，避免 CUDA 检测 |
| 静态库符号未解析 | `--whole-archive` 强制完整包含 |
| `-lcudart` 链接顺序错误 | 统一放入 `add_shflags()` 控制顺序 |

验证结果：Nvidia 和 Iluvatar 平台的 runtime、算子、端到端推理测试全部通过。

---

**项目 #3：AI 聊天机器人**

实现内容：
1. 随机采样：支持 Temperature、Top-K、Top-P、Seed（C API + Python 封装）
2. 聊天服务器（`python/llaisys/server.py`）：HTTP 服务，兼容 OpenAI Chat Completion API（`/v1/chat/completions`），支持流式输出（SSE）和非流式输出
3. 前端 UI（`frontend/`）：Web 界面，支持连续对话、流式显示
4. 会话管理：多会话支持、历史消息编辑 + 分叉重新生成、前缀匹配 KV Cache 池跨会话复用

架构：
```
前端 (HTML/JS) → HTTP → 服务器 (Python) → C API → C++ 推理引擎
                                              ↕
                                        KV Cache Pool
```

---

**项目 #4：多用户推理服务**

实现内容：
1. 请求调度器（`python/llaisys/scheduler.py`）：入口线程 + 调度器 + Worker 执行模式，支持多 Worker、请求队列、超时控制、会话粘性路由 + KV 感知路由
2. 连续批处理：迭代级批处理、Packed Prefill、动态缩批、流式 + 非流式请求均走批量路径
3. 共享模型池（`--shared-model`）：多 Worker 共享一份模型，内存从 N×model_size 降到 1×model_size
4. KV 内存感知流控（`--kv-memory-threshold`）：内存压力超阈值时拒绝新请求（429）

推荐启动参数：
```bash
python -m llaisys.server --model <模型路径> \
  --workers 4 --shared-model \
  --continuous-batching --max-batch-size 8 \
  --kv-aware-routing --kv-memory-threshold 0.85
```

压测结果：

| 参数 | 成功率 | 吞吐 | 平均延迟 |
|------|--------|------|----------|
| total=20, concurrency=2, tokens=16 | 20/20 | 0.18 rps | 11.1s |
| total=12, concurrency=4, tokens=8 | 12/12 | 0.37 rps | 10.2s |

---

**项目 #5：分布式推理**

引入张量并行，将模型分片到多个 GPU 上实现分布式推理，使用 NCCL 通信。

实现内容：

1. 通信层（C API + C++ + NCCL 后端）：
   - `include/llaisys/comm.h` → C API 头文件（函数指针表）
   - `src/device/comm_api.{hpp,cpp}` → C++ dispatcher（#ifdef 条件编译）
   - `src/device/nvidia/nvidia_comm.cu` → NCCL 后端（8 个操作）
   - 支持操作：init、destroy、get_rank、get_size、allreduce、broadcast、send、recv

2. 张量并行（Megatron-style）：Decoder 中每层插入 2 个 AllReduce（`attn_o` 和 `mlp_down` 线性投影后、残差加之前），单 GPU 时零开销

3. 权重切分（`python/llaisys/tensor_parallel.py`）：

   | 权重 | 切分方式 | 说明 |
   |------|----------|------|
   | Q/K/V/gate/up | Column split (dim 0) | 每 rank 获得 nh/tp_size 个 head |
   | attn_o/down | Row split (dim 1) | 输出需 AllReduce 聚合 |
   | embeddings/norms | 复制 | 所有 rank 持有完整副本 |

4. 多进程启动器：Rank 0 生成 NCCL unique ID → 文件 IPC 广播 → 各 rank 加载切分权重 → 分布式推理

验证结果（8×A100-80GB 服务器）：

| 测试 | 结果 |
|------|------|
| 单卡 runtime + 算子 | ✅ 通过 |
| 通信层单元测试 | ✅ 通过 |
| 2 卡 AllReduce | ✅ 通过（SUM = 3.0） |
| 4 卡 AllReduce | ✅ 通过（SUM = 10.0） |
| 8 卡 AllReduce | ⚠️ 超时（显存被其他进程占用） |
| 张量并行推理 | ✅ 通过（2 卡，token 一致） |

## 四、代码架构

```
llaisys/
├── include/llaisys/         # C API 头文件
│   ├── llaisys.h            # 基础类型定义
│   ├── runtime.h            # 运行时 API
│   ├── comm.h               # 通信 API
│   └── models/qwen2.h       # 模型 API
├── src/
│   ├── device/              # 设备抽象层
│   │   ├── cpu/             # CPU 实现
│   │   ├── nvidia/          # CUDA 实现 + NCCL 通信
│   │   └── iluvatar/        # 天数 CoreX 实现
│   ├── ops/                 # 算子（9 个，各含 cpu/nvidia 子目录）
│   ├── models/              # 模型实现（Qwen2 Decoder）
│   └── core/                # 运行时核心（Context/Runtime/Storage）
├── python/llaisys/          # Python 前端
│   ├── server.py            # 聊天服务器
│   ├── scheduler.py         # 请求调度器
│   ├── tensor_parallel.py   # 权重切分
│   └── libllaisys/          # ctypes 绑定
├── frontend/                # Web UI
├── scripts/                 # 工具脚本（启动器、压测）
├── test/                    # 测试文件
└── xmake.lua                # 构建配置
```

## 五、复现流程

### 环境要求

| 依赖 | 版本要求 | 用途 |
|------|----------|------|
| Xmake | >= 2.7 | 构建工具 |
| C++ 编译器 | GCC >= 9 / Clang >= 10 / MSVC 2019+ | 编译后端 |
| Python | >= 3.9 | 前端 + 测试 |
| PyTorch | >= 2.0 | 对比验证（仅测试时需要） |
| CUDA Toolkit | >= 11.0 | GPU 推理（项目 #2 起） |
| NCCL | >= 2.10 | 分布式推理（项目 #5） |

### 步骤 0：克隆仓库 + 下载模型

```bash
git clone https://github.com/KevinSusan/llaisys_tt.git
cd llaisys_tt

# 下载测试模型 DeepSeek-R1-Distill-Qwen-1.5B（约 3GB）
pip install huggingface_hub
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --local-dir ./model
# 国内镜像：HF_ENDPOINT=https://hf-mirror.com huggingface-cli download ...
```

> 以下所有命令中 `./model` 替换为实际模型路径。

---

### 作业 #1-#3 验证（CPU，任意机器）

```bash
# 编译
xmake build

# 安装共享库（Linux）
cp build/linux/x86_64/release/libllaisys.so python/llaisys/libllaisys/
# Windows: copy build\windows\x64\release\llaisys.dll python\llaisys\libllaisys\

# 设置 Python 路径
export PYTHONPATH=$(pwd)/python:$PYTHONPATH

# 作业 #1：张量测试
python test/test_tensor.py

# 作业 #2：CPU 算子测试
python test/test_ops.py

# 作业 #3：CPU 端到端推理（输出应与 PyTorch 完全一致）
python test/test_infer.py --model ./model --test
```

---

### 项目 #2 验证（Nvidia GPU）

设备要求：Nvidia GPU + CUDA Toolkit

```bash
# 编译（开启 Nvidia GPU 支持）
xmake f --nv-gpu=y -c
xmake build
cp build/linux/x86_64/release/libllaisys.so python/llaisys/libllaisys/
export PYTHONPATH=$(pwd)/python:$PYTHONPATH

# GPU 运行时测试
python test/test_runtime.py --device nvidia

# GPU 算子测试（9 个算子）
python test/ops_gpu/run_all.py --device nvidia

# GPU 端到端推理（输出应与 PyTorch 完全一致）
python test/test_infer.py --model ./model --test --device nvidia
```

### 项目 #2 验证（天数 Iluvatar CoreX GPU）

设备要求：天数 Iluvatar CoreX GPU + CoreX SDK

```bash
xmake f --iluvatar-gpu=y -c
xmake build
cp build/linux/x86_64/release/libllaisys.so python/llaisys/libllaisys/
export PYTHONPATH=$(pwd)/python:/usr/local/corex/lib64/python3/dist-packages:$PYTHONPATH

python test/test_runtime.py --device iluvatar
python test/ops_gpu/run_all.py --device iluvatar
python test/test_infer.py --model ./model --test --device iluvatar
```

---

### 项目 #3 验证（聊天机器人）

设备要求：同项目 #2（GPU 推理）

```bash
# 启动聊天服务器
python -m llaisys.server --model ./model --device nvidia

# 在另一个终端测试 API（兼容 OpenAI 格式）
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'

# 或打开浏览器访问 http://localhost:8000 使用 Web UI
```

---

### 项目 #4 验证（多用户推理服务）

设备要求：同项目 #2（GPU 推理）

```bash
# 启动多用户服务（共享模型 + 连续批处理）
python -m llaisys.server --model ./model --device nvidia \
  --workers 2 --shared-model --continuous-batching --max-batch-size 8

# 运行调度器测试
python test/test_scheduler_inmemory.py

# 运行并发压测
python scripts/benchmark_chat_scheduler.py \
  --url http://localhost:8000 --total 12 --concurrency 4 --max-new-tokens 8
```

---

### 项目 #5 验证（分布式推理）

设备要求：多张 Nvidia GPU + NCCL

```bash
# 编译（确保 --nv-gpu=y）
xmake f --nv-gpu=y -c && xmake build
cp build/linux/x86_64/release/libllaisys.so python/llaisys/libllaisys/
export PYTHONPATH=$(pwd)/python:$PYTHONPATH
pip install transformers safetensors

# 通信层单元测试（单卡）
python test/test_comm_api.py --device nvidia

# 多卡 AllReduce 集成测试
python test/test_allreduce.py --nranks 2 --device nvidia
python test/test_allreduce.py --nranks 4 --device nvidia

# 张量并行推理（2 卡）
python scripts/launch_tp.py \
  --model ./model --nranks 2 --device nvidia \
  --prompt "Hello, world" --max-tokens 32
```

## 六、技术亮点

1. **跨平台 CUDA 适配**：通过 kernel 零复制策略，天数 Iluvatar 平台无需修改任何 CUDA 内核代码，直接复用 Nvidia 实现
2. **完整推理服务栈**：从底层 C++ 算子到 HTTP API，全链路自研，兼容 OpenAI API 格式
3. **连续批处理**：迭代级调度 + Packed Prefill + 动态缩批，支持流式和非流式混合请求
4. **Megatron-style 张量并行**：通信层抽象设计，支持 NCCL/IXCCL/MPI 多后端，Decoder 中仅需 2 个 AllReduce/层
5. **KV Cache 复用体系**：前缀匹配 + 跨会话 donor 复用 + 分叉编辑 + 内存感知流控
