# 面试问答（聚焦项目 2 第二平台 MetaX/MACA）

## 1. 你这次交的到底是什么

我这次交的是 LLAISYS 项目 2 的第二平台实现。  
原始分支已经有 CPU 和 NVIDIA 路径，我这次把第二平台从“Metax 设计稿”推进成了“MetaX/MACA 可实际测试”的实现，并在真实沐曦机器上完成了 runtime、ops 和 infer 验证。

## 2. 你为什么选 MetaX/MACA

因为课程要求第二平台，而当前机器就是沐曦环境，SDK、编译器、驱动和 PyTorch 适配都是现成的，能直接做真实验证，不是纸面设计。

## 3. 你是怎么确认这台机器真的是沐曦平台的

我先读了仓库根目录那份沐曦平台说明 PDF，然后用本机工具做了环境识别：

- `mx-smi` 能看到 `MetaX C500`
- `MACA Version` 是 `3.2.1.10`
- 驱动版本是 `3.0.11`
- 编译器是 `mxcc 1.0.0`
- 头文件在 `/opt/maca/include`
- 运行库在 `/opt/maca/lib` 和 `/opt/mxdriver/lib`

所以这个判断不是靠文档猜的，是靠本机 SDK 和驱动直接确认的。

## 4. MetaX 是不是 CUDA 兼容平台

要分两层回答。

第一层，C/C++ SDK 层面不是 CUDA drop-in 兼容。因为它用的是 `mc_runtime` 和 `mcblas`，不是 `cuda_runtime` 和 `cuBLAS`，编译器也是 `mxcc` 不是 `nvcc`。

第二层，PyTorch 使用层面是 CUDA 语义兼容。  
我在本机上验证过 `torch.cuda.is_available()` 是真的，而且 `torch.cuda.get_device_name(0)` 能返回 `MetaX C500`。所以 Hugging Face 对照测试还能继续走 `torch.cuda`。

## 5. 既然 PyTorch 能用 `torch.cuda`，你为什么不直接复用 NVIDIA 后端

因为 PyTorch 的 CUDA 语义兼容，不代表 C++ 后端也能直接拿 CUDA 代码编过。  
如果把 MetaX 硬塞进 `nvidia` 分支，一方面很容易引入条件编译污染，另一方面出问题时也很难判断到底是 NVIDIA 回归还是 MetaX 新问题。

所以我用了独立 `METAX` 后端：

- 保留 `CPU` 和 `NVIDIA`
- 新增 `METAX`
- 构建开关也独立成 `--metax-gpu=y`

这样能最大程度避免改坏原路径。

## 6. 你做了哪些核心改动

核心改动有四块：

1. 设备抽象  
新增 `LLAISYS_DEVICE_METAX`，让运行时和算子调度能识别第二平台。

2. 构建系统  
新增 MetaX 的 xmake 规则，用 `mxcc` 编译 `.cu`，链接 `mcblas` 和 `mcruntime`。

3. runtime  
把 `get_device_count`、`set_device`、`malloc/free`、`memcpy`、`stream` 等接口映射到 `mc*` API。

4. 算子  
把 `add`、`embedding`、`linear`、`rms_norm`、`rope`、`swiglu` 落到 MetaX 设备实现；`argmax` 和 `self_attention` 先用 host fallback 保证链路打通。

## 7. 你为什么允许 host fallback

因为课程当前阶段的目标是把第二平台从设计稿推进到可测试状态。  
如果一开始就要求所有算子都做成高性能设备版本，周期会明显超出这次提交范围。

所以我先做了一个工程上更稳的顺序：

- 先打通 runtime
- 再保证 `test_runtime` 通过
- 再保证 `test_ops` 通过
- 最后保证 `test_infer` 通过

`argmax` 和 `self_attention` 的 host fallback 就是为了先把这条主链路打通。

## 8. `linear` 为什么单独拿出来说

因为 `linear` 是大模型推理里最关键的算子，第二平台成不成立，很大程度取决于它能不能稳定过测试。

我这里没有手写 GEMM，而是直接接了 `mcblasGemmEx`。  
另外我还把数学模式切到了 `MCBLAS_PEDANTIC_MATH`，并让 `float32` 路径使用 `MCBLAS_COMPUTE_32F_PEDANTIC`，这是为了减少和 PyTorch 对照时的微小数值漂移。

## 9. 你遇到的最关键问题是什么

最关键的问题不是写 kernel，而是先判断平台边界。

如果一开始误以为 MetaX 可以直接复用 CUDA 后端，后面会在编译、链接、运行时和数值问题上反复撞墙。  
真正关键的一步，是先确认：

- C++ 层不是 CUDA drop-in
- Python 层是 CUDA 语义兼容

这个判断定下来以后，后面的工程方案才会稳定。

## 10. 还有哪些实际踩坑

### 10.1 xmake 在 root 下默认拒绝运行

本机是 root 环境，`xmake` 会直接报危险并退出。  
解决方式是所有构建命令都显式带：

```bash
XMAKE_ROOT=y
```

### 10.2 当前机器没有现成的本地模型目录

为了先把 `test/test_infer.py` 打通，我选了一个公开的小模型：

- `trl-internal-testing/tiny-Qwen2ForCausalLM-2.5`

它足够小，适合做严格一致性测试，而且 `model_type` 是 `qwen2`，和当前仓库的模型实现完全对齐。

## 11. 你最后跑通了哪些测试

我按下面顺序验证：

```bash
XMAKE_ROOT=y xmake f --metax-gpu=y -cv
XMAKE_ROOT=y xmake -r
XMAKE_ROOT=y xmake install

python test/test_tensor.py
python test/test_runtime.py --device cpu
python test/test_ops.py --device cpu
python test/test_infer.py --device cpu --test --model_id trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 --prompt hi --max_steps 1

python test/test_runtime.py --device metax
python test/test_ops.py --device metax
python test/test_infer.py --device metax --test --model_id trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 --prompt hi --max_steps 1
```

结论是 CPU 基线没被破坏，MetaX 主链路也已经可测。

## 12. 你怎么证明没有改坏 NVIDIA

我采取的是“隔离式接入”：

- 新增 `METAX` 分支，不改写 `NVIDIA` 分支
- 新增 `--metax-gpu=y`，不替换 `--nv-gpu`
- MetaX 代码全部放到独立目录

所以从代码组织上，它不会和 NVIDIA 混成一条路径。  
但要诚实说明的是：当前机器没有 NVIDIA 硬件，所以这次没有做新的 NVIDIA 实机回归。

## 13. 这次提交的边界是什么

这次提交的边界很明确：

- 已完成：第二平台 MetaX/MACA 的可测试实现
- 已完成：CPU 与 MetaX 的 `runtime / ops / infer` 验证
- 未承诺：本次机器上的 NVIDIA 新实机回归
- 未承诺：MetaX 上 `argmax` / `self_attention` 的高性能设备实现
- 未承诺：旧文档里提到的 `TinyLlama/Llama` 完整作业交付

## 14. 如果老师问你这次最体现工程能力的地方是什么

我会强调两点。

第一点是平台判断能力。  
我不是看到 `torch.cuda` 能用就直接假设整个后端也能复用 CUDA，而是把 C++ SDK 层和 Python 语义层分开判断，最后做出独立后端方案。

第二点是落地顺序控制。  
我没有一上来铺开做所有高性能 kernel，而是先用最小可行方案把 `runtime -> ops -> infer` 打通，再把能稳定通过课程测试的链路固定下来。

## 15. 一句话总结

这次我做的不是“补一份第二平台设计稿”，而是把 MetaX/MACA 真正接进了 LLAISYS，并在真实沐曦机器上验证到了可提交、可复现、可测试的状态。
