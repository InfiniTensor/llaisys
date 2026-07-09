# 项目#2 学习型报告：从 0 到 1 集成 CUDA（小白友好 + 弱 C++ 版）

## 1. 先说结论：你在项目二里完成了什么
你已经把 LLAISYS 从“只会 CPU”推进到“能在 NVIDIA GPU 上完整跑推理链路”。

你完成的是一条完整工程链，而不是单点代码：
- Runtime 层：支持 NVIDIA 设备初始化、显存申请、数据拷贝、流同步。
- 算子层：关键算子都能在 `--device nvidia` 下跑通（先用 staging 方案保证正确性）。
- 模型层：`Qwen2` 推理流程能在 nvidia 设备上正常执行。
- 验证层：`build -> install -> test_runtime -> test/ops -> test_infer` 全链路通过。

一句话总结：项目二的价值是“把 GPU 后端这条路打通，并且可复现”。

## 2. 给完全初学者的背景补课

### 2.1 什么是 Runtime API（为什么必须有它）
你可以把 Runtime API 理解为“统一设备操作的遥控器”：
- 不同设备（CPU、NVIDIA）内部实现不同。
- 但上层算子不想关心细节，只想调用统一接口。

所以项目里会有 `runtime_api.hpp` 这种抽象层，然后 CPU/NVIDIA 各自实现。

### 2.2 什么是“端到端可用”
不是“我写了几个 `.cpp` 文件就算完成”，而是下面全部成立：
1. 能编译。
2. 能安装成 Python 可调用动态库。
3. Python 测试能调到 GPU 后端。
4. 算子输出正确。
5. 模型推理也正确。

## 3. 你的核心实现（按模块拆解）

### 3.1 Runtime 层：你做了 NVIDIA API 对接
核心文件：`src/device/nvidia/nvidia_runtime_api.cpp`

你把框架抽象接口映射到了 CUDA 官方 API：
- 设备：`cudaGetDeviceCount`、`cudaSetDevice`、`cudaDeviceSynchronize`
- Stream：`cudaStreamCreate`、`cudaStreamDestroy`、`cudaStreamSynchronize`
- 内存：`cudaMalloc/cudaFree`、`cudaMallocHost/cudaFreeHost`
- 拷贝：`cudaMemcpy`、`cudaMemcpyAsync`

对新手最关键的理解：
- 你的框架不是直接 everywhere 写 CUDA，而是先走统一抽象，再由 NVIDIA 后端实现具体细节。
- 这种设计后面扩展 AMD/其它设备会更容易。

### 3.2 共享 GPU 稳定性：你做了“可用设备筛选”
场景是共享 A100，某些卡可能不可用，直接 `cudaSetDevice` 会失败。

你做的处理是：
- 启动时逐卡探测。
- 仅把“可成功激活”的卡加入可用列表。
- 上层看到的是逻辑设备编号，底层再映射到物理卡。

价值：避免“机器有 8 张卡但你正好选到坏卡”导致测试假失败。

### 3.3 构建系统：你把 NVIDIA 后端接入 xmake
关键文件：`xmake/nvidia.lua`、`xmake.lua`

你完成了：
- 新增 nvidia 目标库并纳入总构建。
- `--nv-gpu=y` 时启用 `ENABLE_NVIDIA_API`。
- Linux 动态库链接增加 `gomp`（OpenMP 运行时）。

为什么这一步重要：
- 很多“代码没错但跑不起来”的问题都在链接阶段。
- 你把“编译通过”和“运行时符号可解析”都兜住了。

### 3.4 算子层：你采用 staging 方案优先保正确
关键思路：
- 数据 D2H（Device 到 Host）
- 复用 CPU 算法
- 结果 H2D 写回

涉及关键算子：
- `add`、`linear`、`argmax`、`embedding`、`rms_norm`、`rope`、`self_attention`、`swiglu`

你这个选择非常工程化：
- 第一阶段先拿到“可跑、正确”。
- 第二阶段再逐个替换成真正 CUDA Kernel 做性能优化。

### 3.5 模型层：你把 `qwen2` 的设备读写改成安全模式
关键文件：`src/llaisys/qwen2.cc`

你新增/使用了安全 helper：
- `zero_tensor`
- `tensor_write_i64`
- `tensor_read_i64`
- `tensor_copy_bytes`

这一步解决了新手常见大坑：
- 在 GPU 上不能像 CPU 一样随便 `memcpy`/解引用设备指针。
- 必须通过 runtime API 做合法的 H2D/D2H/D2D 操作。

## 4. 你踩过并修复的关键问题（学习价值很高）

### 4.1 本机无 CUDA SDK
- 现象：`Cuda SDK not found!`
- 处理：转远端 A100 环境。

### 4.2 远端缺 xmake
- 现象：`xmake: command not found`
- 处理：安装 `~/.local/bin/xmake`，后续固定绝对路径。

### 4.3 动态库 CUDA 注册符号异常
- 现象：`undefined symbol: __cudaRegisterLinkedBinary...`
- 处理：统一为 `cpp + cudart` 路径，移除问题 `.cu` 路线。

### 4.4 OpenMP 链接缺失
- 现象：`omp_get_thread_num` undefined
- 处理：`xmake.lua` 添加 `add_syslinks("gomp")`。

### 4.5 共享卡不可用
- 处理：设备筛选+逻辑映射+固定 `CUDA_VISIBLE_DEVICES`。

## 5. 如何复现你的项目二成果（一步一步）

### 5.1 环境准备
```bash
cd /home/yuanstar/llaisys
export PYTHONPATH=/home/yuanstar/llaisys/python
export CUDA_VISIBLE_DEVICES=2
```

### 5.2 构建安装
```bash
/home/yuanstar/.local/bin/xmake f --nv-gpu=y -cv
/home/yuanstar/.local/bin/xmake -y
/home/yuanstar/.local/bin/xmake install -y
```

### 5.3 运行验证
```bash
python3 test/test_runtime.py --device nvidia

python3 test/ops/add.py --device nvidia
python3 test/ops/linear.py --device nvidia
python3 test/ops/argmax.py --device nvidia
python3 test/ops/embedding.py --device nvidia
python3 test/ops/rms_norm.py --device nvidia
python3 test/ops/rope.py --device nvidia
python3 test/ops/self_attention.py --device nvidia
python3 test/ops/swiglu.py --device nvidia

python3 test/test_infer.py --model /home/yuanstar/models/DeepSeek-R1-Distill-Qwen-1___5B --device nvidia --test --max_steps 8
```

### 5.4 成功标准（你答辩时可以直接说）
- Runtime 测试通过。
- 核心算子测试通过。
- `test_infer` 在 nvidia 下通过，且与参考输出一致。

## 6. 你项目二的能力成长（面向答辩表达）
你不仅“会调 API”，而且已经体现了以下工程能力：
- 抽象层思维：Runtime 统一接口。
- 系统排错能力：从编译、链接、运行时逐层定位。
- 资源受限环境适配：共享 GPU 稳定性处理。
- 交付意识：不仅改代码，还确保复现脚本和验证闭环。

## 7. 面向下一步优化（可选，不影响已完成）
- 把 staging 算子逐步替换为原生 CUDA kernel，重点先做 `linear`、`self_attention`。
- 给关键性能路径增加 benchmark，量化 GPU 加速收益。
- 将“设备筛选 + 健康检查”做成统一工具函数，降低维护成本。

## 8. 项目二一句话复盘
项目二你已经完成到“可提交且可复现”的标准：CUDA 后端集成成功，nvidia 测试链路打通，模型推理在真实 GPU 环境下验证通过。