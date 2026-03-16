# 项目二报告：基于CUDA的推理引擎实现

## 一、 概述

本项目旨在为大语言模型推理框架 `LLAiSYS` 构建底层的 CUDA 算子库。大语言模型（如 Qwen2 系列）的自回归推理过程高度依赖 GPU 的并行计算能力与显存吞吐率。为满足框架在 NVIDIA GPU 上的运行需求，本项目基于 CUDA C++ 实现了模型推理所需的全部核心算子，涵盖 `Add`、`SwiGLU`、`RMSNorm`、`Linear`、`Embedding`、`RoPE`、`Argmax` 以及 `Self-Attention`。

该算子库原生支持 `FP32`、`FP16` 及 `BF16` 数据类型。所有算子均通过了与 PyTorch 原生实现的精度对比测试，并在端到端推理验证中实现了输出 Token 序列的 100% 精度对齐，为上层推理服务提供了可靠的算力基础。

## 二、 运行环境

- **硬件平台**：NVIDIA GPU（测试环境基于 A100 Tensor Core GPU x8）
- **操作系统**：Linux / Windows 跨平台支持
- **核心语言**：C++ 17, CUDA C++, Python 3.10+
- **构建系统**：Xmake
- **依赖库**：CUDA Toolkit, cuBLAS (NVIDIA Basic Linear Algebra Subprograms)
- **验证基准**：PyTorch 2.x

## 三、 核心架构与具体实现

### 3.1 算子构建与链接架构

在算子库构建初期，C++ 静态库之间的循环依赖导致了 CUDA 宏注入失败与符号丢失（Undefined Symbol）问题。本项目通过重构 `xmake.lua`，将算子的编译与链接权限上移至动态链接库（`libllaisys.so` / `.dll`），实现了多级依赖环境下的 CUDA 文件安全编译。

**核心构建配置示例 (`xmake.lua`)：**

```lua
target("llaisys")
    set_kind("shared")
    add_deps("llaisys-utils", "llaisys-device", "llaisys-core", "llaisys-tensor", "llaisys-ops", "llaisys-models") 

    if has_config("nv-gpu") then
        add_rules("cuda")
        -- 将算子的 CUDA 实现统一交由拥有一切依赖的上层动态库编译
        add_files("src/ops/*/nvidia/*.cpp", "src/ops/*/nvidia/*.cu")
    end
target_end()
```

### 3.2 核心算子实现细节

#### 1. 基础并行计算 (Add & SwiGLU)

对于 Element-wise（逐元素）操作，采用网格跨步循环（Grid-Stride Loop）以适配任意长度的 Tensor。在处理 `FP16` 和 `BF16` 类型时，通过寄存器级别的类型转换（如 `__half2float`），将数据提升至单精度进行非线性计算，以保证数值稳定性。

#### 2. 并行规约算子 (RMSNorm & Argmax)

规约（Reduction）操作是典型的显存带宽瓶颈。

- **RMSNorm**：采用 Block 级别的并行计算。为每个 Token 分配一个 Thread Block，利用 `__shared__` 内存进行块内规约求平方和，并使用 CUDA 硬件指令 `rsqrtf` 计算均方根倒数。
- **Argmax**：为应对输出层巨大的词表维度，通过维护线程局部极值（`local_max`）与局部索引（`local_idx`），随后在单 Block 内通过共享内存规约出全局极值。

#### 3. 矩阵乘法 (Linear)

大语言模型的全连接层运算由 `cuBLAS` 库接管，以充分利用 GPU 的 Tensor Cores。由于 C/C++ 采用行优先（Row-Major）存储，而 cuBLAS 基于列优先（Column-Major），系统利用转置等价性公式 $(AB^T)^T = BA^T$ 设置 `CUBLAS_OP_T` 参数，实现了零拷贝的矩阵乘法，随后通过轻量级 Kernel 注入偏置项（Bias）。

#### 4. 显存寻址与相对位置编码 (Embedding & RoPE)

在索引寻址类算子中，系统严格规范了数据类型对齐与内存步长：

- **数据类型对齐**：确保 `index` / `pos_ids` 强制使用 `int64_t` 指针进行解引用，避免因 Python 端与 C++ 端类型错位导致的内存越界。
- **RoPE 内存步长**：严格匹配标准 PyTorch 的 RoPE 语义，将特征维度切分为前后两半（`half_dim`），对间隔 `half_dim` 的元素对执行复数旋转。

**RoPE 核心寻址逻辑片段：**

```c++
// 计算内存偏移，前一半与后一半组合为一对复数
size_t idx_a = seq_idx * (nhead * head_dim) + head_idx * head_dim + pair_idx;
size_t idx_b = idx_a + half_dim;

float x0 = in[idx_a];
float x1 = in[idx_b];
out[idx_a] = x0 * cos_m - x1 * sin_m;
out[idx_b] = x1 * cos_m + x0 * sin_m;
```

#### 5. 分组查询自注意力 (Self-Attention with GQA)

该算子采用分块并行（Block-level Parallelism）设计，Grid 维度设定为 `[seqlen, nhead]`，使每个 Block 独立处理单个 Query 向量。

- **动态共享内存**：在 Block 内部申请长度为 `total_len` 的动态共享内存 `extern __shared__ float scores[]`。
- **Softmax 融合与 Causal Mask**：计算点积时，通过索引比对将未来位置的分数置为负无穷（`-1e20f`）。点积完成后，直接在共享内存中就地执行 Softmax 操作并与 Value 矩阵进行加权求和，避免了中间结果落入全局显存。

## 四、 构建与测试

### 4.1 项目构建说明

本项目依赖 `xmake` 工具进行工程管理与编译。构建全量带有 NVIDIA GPU 支持的动态链接库与 Python 包包，执行以下标准流程：

Bash

```
# 清理构建缓存并重新配置 GPU 编译选项
xmake clean -a
xmake f -c --nv-gpu=y
# 编译并生成共享库
xmake -r install
# 将生成的 C++ 库注册至 Python 环境
pip install ./python/
```

### 4.2 算子单元测试

框架针对各算子实现了独立的 Python 测试脚本。测试脚本基于 `ctypes` 调用生成的 `libllaisys.so` 接口，并采用 PyTorch 同等运算作为对照组，利用 `torch.allclose` 验证 `atol` 与 `rtol`。

```Bash
python test/ops/add.py --device nvidia
python test/ops/swiglu.py --device nvidia
python test/ops/rms_norm.py --device nvidia
python test/ops/rope.py --device nvidia
python test/ops/self_attention.py --device nvidia
```

所有算子均能稳定通过全精度（F32）、半精度（F16/BF16）的边界测试与精度校验。

### 4.3 端到端推理验证

在单算子验证通过的基础上，对 Hugging Face 开源模型 `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` 进行自回归生成测试，对比原生 PyTorch 推理结果与 LLAiSYS 推理结果。

**验证指令与结果示例：**

```Bash
python test/test_infer.py --device nvidia
```

- **输出比对**：LLAiSYS 输出的 Token ID 序列与 PyTorch 生成的 Token 序列一致。
- **功能验证**：模型能够成功载入权重配置，正确处理 KV Cache 状态，并连续生成逻辑连贯的文本。

## 五、 结论

本项目成功在 LLAiSYS 框架中构建了底层 CUDA 算子生态。通过解决多级依赖构建、张量内存排布、类型边界控制等关键工程问题，实现了大语言模型所需全套核心算子的高效 GPU 并行计算。