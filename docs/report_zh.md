# LLAISYS 课程作业与项目实现报告

## 1. 提交结论

本次提交按完整课程交付组织，覆盖：

- Assignment #1：Tensor
- Assignment #2：Operators
- Assignment #3：Large Language Model Inference
- Project #1：CPU 优化
- Project #2：第二平台 MetaX/MACA
- Project #3：聊天服务
- Project #6：支持新模型

最终实现的核心结论是：

- 已完成 Tensor、核心算子与 Qwen2 推理链路实现
- 已完成 CPU 热点算子优化
- 已完成第二平台 MetaX/MACA 接入与实机验证
- 已完成聊天服务接口与流式返回链路
- 已完成 `Llama/TinyLlama` 新模型接入所需的后端与 Python 包装

从整体上看，当前仓库已经覆盖了从底层 Tensor、算子、推理流程，到平台适配和上层聊天接口的一条完整实现路径。

## 2. Assignment #1：Tensor

Assignment #1 的目标是实现 LLAISYS 最基础的数据结构 Tensor。  
这一部分围绕张量元信息、存储布局与视图变换展开。

完成内容包括：

- `load`
- `isContiguous`
- `view`
- `permute`
- `slice`

这一阶段的核心收获是：

- 理解了 `shape`、`stride`、`offset` 与 `storage` 的关系
- 理解了视图变换和真实数据拷贝的区别
- 为后续所有算子实现建立了统一的 Tensor 抽象基础

当前通过：

```bash
python test/test_tensor.py
```

## 3. Assignment #2：Operators

Assignment #2 的目标是在 CPU 上补齐推理链路所需的关键算子。

已实现的主要算子包括：

- `argmax`
- `embedding`
- `linear`
- `rms_norm`
- `rope`
- `self_attention`
- `swiglu`

实现时重点保证：

- 支持 `float32`、`float16`、`bfloat16`
- 输入输出张量 shape 约束正确
- Python 测试入口与 C/C++ 实现链路打通

当前通过：

```bash
python test/test_runtime.py --device cpu
python test/test_ops.py --device cpu
```

## 4. Assignment #3：Large Language Model Inference

Assignment #3 的目标是基于前面的 Tensor 与算子，完成一个真正可运行的 Qwen2 推理链路。

完成内容包括：

- Qwen2 配置解析
- 权重装载与后端权重槽位映射
- 推理主链路组织
- 增量生成接口
- 与 Hugging Face 的 token 级对照验证

这一部分重点解决的问题包括：

- 权重文件如何映射到后端固定结构
- 推理阶段为什么要做增量解码
- 如何通过 token 级对照判断实现正确性

当前在本地 CPU 环境通过：

```bash
python test/test_infer.py --device cpu --test --model models/DeepSeek-R1-Distill-Qwen-1.5B --prompt hi --max_steps 1
```

## 5. Project #1：CPU 优化

Project #1 的目标是在已有正确实现基础上，对 CPU 路径做热点算子优化。

完成内容包括：

- 在 CPU 构建规则中启用 OpenMP
- 对 `linear`、`embedding`、`rms_norm`、`rope` 等热点算子进行并行优化
- 保持接口与功能不变，优先保证正确性

其中 `linear` 是最核心的优化点，因为它在推理阶段调用频繁且计算量大，容易成为 CPU 路径的主要瓶颈。

这一部分的工程策略不是追求极限底层微优化，而是采用课程项目里更合理、可解释、可验证的方式：

- 分块循环
- 按行或按 token 并行
- 基于 OpenMP 提升 CPU 多核利用率

## 6. Project #2：第二平台 MetaX/MACA

Project #2 的目标是支持除 CPU / NVIDIA 之外的第二个平台。  
本次最终选择的平台是沐曦 `MetaX/MACA`。

### 6.1 平台判断

在接入前，首先要回答 MetaX 是否能直接复用原有 CUDA 后端。  
实际验证后结论是：

- 在 C/C++ SDK 层面，MetaX 不是 CUDA drop-in 兼容平台
- 在 Python / PyTorch 层面，MetaX 保留了 `torch.cuda` 语义

因此最终采用的策略是：

- C/C++ 后端新增独立 `METAX` 分支
- Python 对照测试仍然复用 `torch.cuda`

### 6.2 主要实现

完成的核心工作包括：

- 在设备枚举中新增 `METAX`
- 在 `xmake.lua` 中新增 `--metax-gpu=y`
- 新增 `xmake/metax.lua`
- 接入 MetaX runtime：设备、stream、内存分配、同步/异步拷贝
- 接入 MetaX 算子路径

当前 MetaX 侧的算子策略为：

- `add`、`embedding`、`rms_norm`、`rope`、`swiglu`：MetaX kernel
- `linear`：`mcblasGemmEx` + bias kernel
- `argmax`、`self_attention`：host fallback

### 6.3 实测结果

在真实沐曦机器上完成如下验证：

```bash
XMAKE_ROOT=y xmake f --metax-gpu=y -cv
XMAKE_ROOT=y xmake -r
XMAKE_ROOT=y xmake install

python test/test_runtime.py --device metax
python test/test_ops.py --device metax
python test/test_infer.py --device metax --test --model_id trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 --prompt hi --max_steps 1
```

验证结论为：

- `runtime` 通过
- `ops` 通过
- `infer` 中 Hugging Face 与 LLAISYS token 级严格一致

这说明第二平台已经从设计说明推进到真实可测试状态。

## 7. Project #3：聊天服务

Project #3 的目标是让系统从“能生成 token”走向“能以聊天形式对外服务”。

完成内容包括：

- 采样参数链路打通
- 基于 FastAPI 实现聊天服务接口
- 基于 SSE 实现流式返回
- 补齐 CLI 交互入口

这一部分的关键点包括：

- 模型生成参数如何从接口层传到后端
- 流式输出如何逐步返回新 token
- 聊天历史如何组织成模型输入 prompt

本地 CPU 环境下已完成最小接口验证：

```bash
PYTHONPATH=python python -m llaisys.chat.server --model models/DeepSeek-R1-Distill-Qwen-1.5B --device cpu --host 127.0.0.1 --port 8011
curl --noproxy '*' -s http://127.0.0.1:8011/health
curl --noproxy '*' -s -X POST http://127.0.0.1:8011/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"你好"}],"stream":false,"max_tokens":8}'
```

当前已确认：

- `/health` 可正常返回
- `POST /v1/chat/completions` 可正常返回非流式结果

## 8. Project #6：支持新模型

Project #6 的目标是在作业默认使用的 Qwen2 之外，再支持另一种模型类型。

本次完成内容包括：

- 新增 `Llama` 对应的 C/C++ 后端模型包装
- 新增 Python 侧 `Llama` 包装类
- 复用 `DecoderOnlyModel` 的通用权重装载、推理与流式生成主链路
- 在 `load_model` 中基于 `config.json` 的 `model_type` 自动分发 `Qwen2` 或 `Llama`

这一部分的核心意义是：

- 把“只能跑一种模型”推进到“同一套推理框架可支持不同模型类型”
- 让权重装载、推理与采样接口尽量复用，减少模型分支重复实现

当前新模型路径的推荐验证入口为：

```bash
python test/test_infer.py --device cpu --test --model /path/to/local/llama_or_tinyllama_model --prompt hi --max_steps 1
```

要求：

- 模型目录中存在 `config.json`
- `config.json` 中 `model_type` 为 `llama`
- 使用当前仓库最新构建结果

## 9. 验证环境与边界

### 9.1 本地 CPU 开发环境

- Python：`3.12.3`
- xmake：`v3.0.7+20260308`
- 本地模型目录：`models/DeepSeek-R1-Distill-Qwen-1.5B`

### 9.2 沐曦 MetaX 验证环境

- GPU：`MetaX C500`
- `mx-smi`：`2.2.9`
- `MACA`：`3.2.1.10`
- 驱动：`3.0.11`
- 编译器：`mxcc 1.0.0`
- Python：`3.10.10`
- PyTorch：`2.6.0+metax3.2.1.3`
- xmake：`v2.8.7+20240401`

### 9.3 验证边界说明

- Assignment #1/#2/#3 与 Project #1/#3/#6 以本地 CPU 路径验证为主
- Project #2 的结论来自真实沐曦机器
- 当前机器没有 NVIDIA 硬件，因此没有新增 `--device nvidia` 的实机回归数据
- 当前推理验证以 `Qwen2` 为主；Project #6 提供 `Llama/TinyLlama` 新模型接入与本地模型目录验证入口

## 10. 问题与处理

### 10.1 Tensor 视图与真实拷贝容易混淆

这一问题主要出现在 `view / permute / slice` 的实现阶段。  
最终通过严格区分 stride 变化和 storage 拷贝逻辑解决。

### 10.2 算子正确性与系统链路正确性不是一回事

单个算子通过测试，并不等于完整推理链路一定正确。  
因此在 Assignment #3 中还需要通过 `infer` 的 token 级对照做最终闭环验证。

### 10.3 第二平台不是 CUDA 直替

这是 Project #2 中最关键的判断问题。  
如果误判成“只要替换宏就能跑”，后续实现很容易失控。最终通过新增独立 MetaX 后端解决。

### 10.4 root 环境下 xmake 默认拒绝运行

在沐曦平台构建时，需要显式设置：

```bash
XMAKE_ROOT=y
```

否则构建会直接被阻止。

## 11. 已知限制

- `argmax` 与 `self_attention` 在 MetaX 侧仍为 host fallback
- 当前推理验证以 `Qwen2` 为主；`Llama/TinyLlama` 路径以代码接入和本地模型目录验证入口为主
- 第二平台已实跑，但不额外展开未在当前机器上重复验证的 CPU / NVIDIA 结果

## 12. 提交边界说明

为保持课程提交 PR 干净，本次仓库提交只保留：

- Assignment / Project 对应实现代码
- 构建、测试与 Python 桥接相关改动
- 正式提交文档

未纳入提交的内容包括：

- 本地学习材料
- 复试问答、讲稿与简历草稿
- 外部平台说明 PDF
- 临时排障文档

## 13. 总结

通过本次课程实践，已经完成从 Tensor、算子、推理链路，到 CPU 优化、第二平台接入、聊天服务实现以及新模型支持的一整条实现路径。

这个项目的最大收获不是“会调用模型”，而是：

- 真正理解了大模型推理系统的底层组成
- 学会了从数据结构、算子、模型链路、平台适配到服务接口的系统化实现方式
- 能够把课程要求的多个模块合并成一个完整、可复现、可提交的系统工程交付
