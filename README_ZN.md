# 欢迎使用 LLAISYS

<p align="center">
<a href="README.md" target="README.md">English</a> ｜
<a href="README_ZN.md" target="README_ZN.md">中文</a>
</p>

## 简介

LLAISYS（Let's Learn AI SYStem）是一个教育项目，旨在为新手和未来的 AI 工程师提供一个从零开始构建 AI 系统的学习平台。LLAISYS 包含多个帮助学生学习和构建基础模块的作业，以及面向通过考核并获准晋级学员的项目阶段。LLAISYS使用C++作为系统后端的主要编程语言，并编译成共享库，提供C语言API。前端代码使用Python编写，调用这些API以提供更便捷的测试和与其他架构（如PyTorch）的交互。

### 项目结构概览

- `\include`：包含所有定义共享库提供的C API的头文件的目录。（函数声明以`__export`开头）

- `\src`：C++源文件。
  - `\src\llaisys`包含头文件中定义的所有直接实现，并遵循与`\include`相同的目录结构。这也是C++代码的边界。
  - 其他目录包含不同模块的实际实现。

- `xmake.lua`：llaisys后端的构建规则。`\xmake`目录包含不同设备的子xmake文件。例如，将来可以在目录中添加`nvidia.lua`来支持CUDA。

- `\python`：Python源文件。
  - `\python\llaisys\libllaisys`包含llaisys API的所有ctypes封装函数。它基本上与C头文件的结构相匹配。
  - `\python\llaisys`包含ctypes函数的Python包装器，使包更符合Python风格。

- `\test`：导入llaisys python包的Python测试文件。

## 作业 #0：入门

### 任务-0.1 安装必备组件

- 编译工具：[Xmake](https://xmake.io/)
- C++编译器：MSVC（Windows）或Clang或GCC
- Python >= 3.9（PyTorch、Transformers等）
- Clang-Format-16（可选）：用于格式化C++代码。

### 任务-0.2 Fork并构建LLAISYS

- Fork LLAISYS仓库并克隆到本地机器。支持Windows和Linux。

- 编译和安装

  ```bash
  # 编译c++代码
  xmake
  # 安装llaisys共享库
  xmake install
  # 安装llaisys python包
  pip install ./python/
  ```

- Github自动测试

  LLAISYS使用Github Actions在每次推送和拉取请求时运行自动化测试。你可以在仓库页面上看到测试结果。完成所有作业任务后，所有测试都应该通过。

### 任务-0.3 首次运行LLAISYS

- 运行cpu运行时测试

  ```bash
  python test/test_runtime.py --device cpu
  ```

  你应该看到测试通过。

### 任务-0.4 下载测试模型

- 我们用于作业的模型是[DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)。

- 使用PyTorch运行模型推理测试

  ```bash
  python test/test_infer.py --model [dir_path/to/model]
  ```

  你可以看到PyTorch能够加载模型并使用示例输入执行推理。你可以调试进入`transformers`库代码来深入查看并了解其内部运作原理。现在，你的代码还无法执行任何操作，但在后续的作业中，你将构建一个能够实现相同功能的系统。

## 作业 #1：张量

张量是表示多维数据的数据结构。它是LLAISYS和大多数AI框架（如PyTorch）的基本构建单元。在这个作业中，你将学习如何实现一个基本的张量类。

张量对象具有以下字段：

- `storage`：指向存储张量数据的内存块的共享指针。它可以被多个张量共享。有关更多详细信息，请查看storage类。
- `offset`：张量在存储中的起始索引（以字节为单位）。
- `meta`：描述张量形状、数据类型和步长的元数据。

实现`src/tensor/tensor.hpp`中定义的以下函数：

### 任务-1.1

```c++
void load(const void *src);
```

将主机（cpu）数据加载到张量（可以在设备上）。查看构造函数了解如何获取当前设备上下文的运行时API，并执行从主机到设备的内存复制。

### 任务-1.2

```c++
bool isContiguous() const; 
```

检查张量的形状和步长，判断它在内存中是否连续。

### 任务-1.3

```c++
tensor_t view(const std::vector<size_t> &shape) const;
```

创建一个新张量，通过拆分或合并原始维度将原始张量重塑为给定形状。不涉及数据传输。例如，通过合并最后两个维度，将形状为(2, 3, 5)的张量更改为(2, 15)。

这个函数不是简单地改变张量的形状那么简单，尽管测试会通过。如果新视图与原始张量不兼容，它应该引发错误。想想一个形状为(2, 3, 5)、步长为(30, 10, 1)的张量。你还能在不传输数据的情况下将其重塑为(2, 15)吗？

### 任务-1.4

```c++
tensor_t permute(const std::vector<size_t> &order) const;
```

创建一个新张量，改变原始张量维度的顺序。转置可以通过这个函数实现，而无需移动数据。

### 任务-1.5

```c++
tensor_t slice(size_t dim, size_t start, size_t end) const;
```

创建一个新张量，沿给定维度，start（包含）和end（不包含）索引对原始张量进行切片操作。

### 任务-1.6

运行张量测试。

```bash
python test/test_tensor.py
```

你应该看到所有测试都通过了。提交并推送你的更改。你应该看到作业#1的自动测试通过了。

## 作业 #2：算子

在这个作业中，你将实现以下算子的cpu版本：

- argmax
- embedding
- linear
- rms_norm
- rope
- self_attention
- swiglu

阅读`src/ops/add/`中的代码，了解"add"算子是如何实现的。确保你理解算子代码是如何组织、编译、链接以及暴露给Python前端的。**你的算子应该至少支持Float32、Float16和BFloat16数据类型**。`src/utils/`中提供了一个用于简单类型转换的辅助函数。所有python测试都在`test/ops`中，你的实现应该至少通过这些测试。首先尝试运行"add"算子的测试脚本。

### 任务-2.1 Argmax

```c++
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals);
```

获取张量`vals`的最大值及其索引，并分别存储在`max_val`和`max_idx`中。你暂时可以假设`vals`是一个1D张量，`max_idx`和`max_val`都是包含单个元素的1D张量（这意味着保留了`vals`的维度）。

完成实现后，你应该能够通过`test/ops/argmax.py`中的测试用例。

### 任务-2.2 Embedding

```c++
void embedding(tensor_t out, tensor_t index, tensor_t weight);
```

从`weight`（2-D）中复制`index`（1-D）中的行到`output`（2-D）。`index`必须是Int64类型（PyTorch中int的默认数据类型）。

完成实现后，你应该能够通过`test/ops/embedding.py`中的测试用例。

### 任务-2.3 Linear

```c++
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias);
```

计算以下内容：

$$
Y = xW^T + b
$$

- `out`：输出 $Y$ 。你暂时可以假设输出是一个2D连续张量，不涉及广播。
- `input`：输入 $X$ 。你暂时可以假设输入是一个2D连续张量，不涉及广播。
- `weight`：权重 $W$ 。2D连续张量。注意权重张量没有转置。你需要在计算过程中处理这个问题。
- `bias`（可选）：偏置 $b$ 。1D张量。你需要支持不提供偏置的情况。

完成实现后，你应该能够通过`test/ops/linear.py`中的测试用例。

### 任务-2.4 RMS Normalization

```c++
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps);
```

为每一行计算以下内容：

$$
Y_i = \frac{W_i \times  X_i}{\sqrt{\frac{1}{d}(\sum_{j=1}^d X_j^2) + \epsilon}}
$$

- `out`：输出 $Y$ 。你暂时可以假设输出是一个2D连续张量，不涉及广播。
- `input`：输入 $X$ 。你暂时可以假设输入是一个2D连续张量，不涉及广播。标准化沿输入张量的最后一个维度（即每一行，长度为 $d$ ）执行。
- `weight`：权重 $W$ 。1D张量，与输入张量的一行长度相同。
- `eps`：小值 $\epsilon$ 以避免除以零。

完成实现后，你应该能够通过`test/ops/rms_norm.py`中的测试用例。

### 任务-2.5 旋转位置编码（RoPE）

```c++
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta);
```

为输入张量`in`的每个向量（这些向量与 pos_ids 中的位置 id 相对应）计算以下内容：

设 $\mathbf{x}_i = [\mathbf{a}_i, \mathbf{b}_i] \in \mathbb{R}^d$ 为输入向量， $\mathbf{y}_i = [\mathbf{a}'_i, \mathbf{b}'_i] \in \mathbb{R}^d$ 为索引 $i$ 处的输出向量，其中 $\mathbf{a}_i, \mathbf{b}_i,\mathbf{a}'_i, \mathbf{b}'_i \in \mathbb{R}^{d/2}$ 。

设 $\theta$ 为固定基数（例如 $\theta = 10000$）， $j = 0, 1, \ldots, d/2 - 1$。

设 $p_i \in \mathbb{N}$ 是输入索引i处token的位置id。

那么RoPE的角度为 $\phi_{i,j} = \frac{p_i}{\theta^{2j/d}}$

输出向量 $\mathbf{y}_i = [\mathbf{a}'_i, \mathbf{b}'_i]$ 计算如下：

$$a_{i,j}' = a_{i,j} \cos(\phi_{i,j}) - b_{i,j} \sin(\phi_{i,j})$$

$$b_{i,j}' = b_{i,j} \cos(\phi_{i,j}) + a_{i,j} \sin(\phi_{i,j})$$

- `out`：结果**q**或**k**张量。形状应该是 [seqlen, nhead, d] 或 [seqlen, nkvhead, d]。你暂时可以假设张量是连续的。
- `in`：原始**q**或**k**张量。形状应该是 [seqlen, nhead, d] 或 [seqlen, nkvhead, d]。你暂时可以假设张量是连续的。
- `pos_ids`：输入序列中每个token的位置id（整个上下文中的索引）。形状应该是 [seqlen,]，dtype应该是int64。
- `theta`：频率向量的基值。

完成实现后，你应该能够通过`test/ops/rope.py`中的测试用例。

### 任务-2.6 自注意力（self-attention）

```c++
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale);
```

为查询张量`q`、键张量`k`和值张量`v`计算自注意力。如果需要，你应该在进行此计算之前连接kvcache张量。

$$
A = Q K^\top * scale \\
$$

$$
Y = \mathrm{causalsoftmax}(A) \cdot V \\
$$

- `attn_val`：结果注意力值张量。形状应该是[seqlen, nhead, dv]。你暂时可以假设张量是连续的。
- `q`：查询张量。形状应该是 [seqlen, nhead, d]。你暂时可以假设张量是连续的。
- `k`：键张量。形状应该是 [total_len, nkvhead, d]。你暂时可以假设张量是连续的。
- `v`：值张量。形状应该是 [total_len, nkvhead, dv]。你暂时可以假设张量是连续的。
- `scale`：缩放因子。在大多数情况下取值为 $\frac{1}{\sqrt{d}}$ 。

完成实现后，你应该能够通过`test/ops/self_attention.py`中的测试用例。

### 任务-2.7 SwiGLU

```c++
void swiglu(tensor_t out, tensor_t gate, tensor_t up);
```

这是一个逐元素函数，计算以下内容：

$$
out_{i} = up_{i} \circ \frac { gate_{i}}{1 + e^{-gate_{i}}}
$$

`out`、`up`和`gate`是具有相同形状 [seqlen, intermediate_size] 的2D连续张量。

完成实现后，你应该能够通过`test/ops/swiglu.py`中的测试用例。

### 任务-2.8

运行算子测试。

```bash
python test/test_ops.py
```

你应该看到所有测试都通过了。提交并推送你的更改。你应该看到作业#2的自动测试通过了。

## 作业 #3：大语言模型推理

终于，是时候用LLAISYS实现文本生成了。

- 在`test/test_infer.py`中，你的实现应该能够使用argmax采样生成与PyTorch相同的文本。我们用于此作业的模型是[DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)。

- 你的实现的python包装器在`python/llaisys/models/qwen2.py`中。你不允许在这里使用任何基于python的框架（如PyTorch）实现你的模型推理逻辑。相反，你需要在LLAISYS后端用C/C++实现模型。脚本加载safetensors文件中的每个张量，你需要从它们加载数据到你的模型后端。

- 在`include/llaisys/models/qwen2.h`中，为你定义了一个原型。你可以随意修改代码，但你应该至少提供模型创建、销毁、数据加载和推理的基本API。在`src/llaisys/`中实现你的C API，并像`src/`中的其他模块一样组织你的C++代码。记得在`xmake.lua`中定义编译过程。

- 在`python/llaisys/libllaisys/`中，为你的C API定义ctypes包装函数。使用你的包装函数实现`python/llaisys/models/qwen2.py`。

- 你需要实现 KV-Cache 功能，否则模型推理速度会过慢。

- 调试直到你的模型工作。利用张量的`debug`函数打印张量数据。它允许你在模型推理期间将任何张量的数据与PyTorch进行比较。

完成实现后，你可以运行以下命令来测试你的模型：

```bash
python test/test_infer.py --model [dir_path/to/model] --test
```

提交并推送你的更改。你应该看到作业#3的自动测试通过了。

## 作业 #4：在 LLAISYS 中集成 CUDA，适配两款CUDA或类CUDA平台（以下统称 CUDA）

需要选择 Nvidia、天数、摩尔、沐曦中的至少两款平台。

本次训练营提供了以上四种平台的算力，可以在官方进行申请算力，并用 CUDA 加速模型推理。在动手前，先深入理解 LLAISYS 框架。

事实上，LLAISYS 是一个支持同构硬件的框架。使用时，每个线程会创建一个线程唯一的 **Context** 对象，管理该线程使用的所有设备 **Runtime**。**Runtime** 对象是设备的资源管理器，**Context** 会为每个设备（以延迟初始化的方式）创建唯一的 **Runtime**。你可以用 ``setDevice`` 在不同设备间切换，每个线程同一时间只会激活一个设备。详情见 ``src/core/context.hpp``。

### 实现 CUDA Runtime API

每个 **Runtime** 对象都会初始化一组通用的 **Runtime API**。你需要实现 CUDA 版本的 API。参考 ``src/device/cpu/cpu_runtime_api.cpp`` 看 CPU 的实现方式，查阅 [`CUDA Runtime 文档`](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html) 找到对应 API。

在 ``src/device/runtime_api.hpp`` 中，``nvidia::getRuntimeAPI()`` 被 ``ENABLE_NVIDIA_API`` 宏保护：

```c++
#ifdef ENABLE_NVIDIA_API
namespace nvidia {
const LlaisysRuntimeAPI *getRuntimeAPI();
}
#endif
```

该宏的定义在 ``xmake.lua`` 中，用于开关 CUDA 支持。若关闭，CUDA 代码不会被编译。你需要在 ``xmake/`` 下新建 ``nvidia.lua``，配置编译流程（参考 ``cpu.lua``）。查阅资料学习如何用 Xmake 配置。

完成 CUDA Runtime API 后，用 ``--nv-gpu=y`` 打开 CUDA 支持并重新编译，运行测试：

```bash
xmake f --nv-gpu=y -cv
xmake
xmake install
python test/test_runtime.py --device nvidia
```

### 实现 CUDA 算子

在每个算子目录下新建 ``nvidia/`` 子目录，写 CUDA 版本实现。参考 ``src/ops/add/op.cpp`` 看如何包含 CUDA 实现。别忘了在 xmake 文件中定义编译流程。用 ``--device nvidia`` 参数运行测试。

你可以使用 cuBLAS、cuDNN 等 CUDA 库来加速算子，额外的设备资源可以放在 `src/device/nvidia/nvidia_resource.cu`。

最后,修改模型代码，支持 CUDA 推理：

```bash
python test/test_infer.py --model [dir_path/to/model] --test --device nvidia
```

提交并推送你的更改。你应该看到作业#4的自动测试通过了。

## 项目阶段：参与 InfiniLM 推理引擎开发

只有通过作业阶段考核并获准晋级的学员，才能进入项目阶段。

所有项目均须基于我们的推理引擎 [InfiniLM](https://github.com/InfiniTensor/InfiniLM) 完成。开始开发前，应与导师确认项目范围；项目成果应对 InfiniLM 具有实际价值，并尽量达到可合入上游的质量。项目将综合考察正确性、工程质量、测试、文档、结果可复现性和实际收益；不同方向的预期工作量会根据课题复杂度调整。

学员可以选择以下项目方向：

### 项目 #1：新模型与新架构适配

在 InfiniLM 中支持新的模型。评判将综合考虑模型复杂度、引入的可复用基础设施，以及实现和测试的完整程度。对于需要引入 MLA、MTP、MoE、NSA、Mamba、RWKV、UltraMem、Titans、MiniMax 等新架构或机制的模型，其工作量与几乎完全复用现有实现的模型变体将区别评估。

### 项目 #2：性能优化

优化 InfiniLM 的纯离线推理性能、服务性能，或同时优化两者。可选工作包括算子与内核优化、模型执行优化、访存优化、通信优化等。评判依据包括端到端性能收益及其可复现性、正确性、覆盖的工作负载，以及能够享受到优化的平台数量和平台价值。

### 项目 #3：推理功能与服务能力完善

完善 InfiniLM 的推理功能与服务能力，例如流式输出、接口兼容、结构化输出、服务可观测性和诊断工具等。评判依据包括设计与实现的完整性、易用性、兼容性、测试和文档。

### 项目 #4：量化与低精度推理

增加或改进权重量化、激活量化、KV Cache 量化、混合精度执行或新的低精度数据格式支持。评判重点包括精度、性能和内存收益、硬件覆盖范围、易用性，以及测试和性能数据的完整程度。

### 项目 #5：可靠性与工程工具

通过性能评测与回归基础设施、性能分析与追踪工具、模型转换与校验工具、兼容性测试或错误诊断等工作，提升 InfiniLM 的可靠性和开发效率。评判依据包括解决实际问题的范围、可维护性、平台覆盖，以及对开发或调试流程带来的可度量改进。

学员也可以提出其他项目方向，但须提前获得导师认可，并针对 InfiniLM 的真实问题给出边界清晰、可度量的交付目标。
