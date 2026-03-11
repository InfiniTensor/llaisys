# LLAISYS 作业提交总览

## 1. 当前提交范围

当前仓库基于 `checkpoint/nvidia-done`，本次新增并已落地的内容是：

- 项目 2 第二平台：MetaX/MACA
- 目标状态：从“设计稿”推进到“真实机器上可测试”
- 约束：不改坏现有 `CPU + NVIDIA` 路径

当前文档全部按这个范围组织，不再沿用旧版“Metax 仅设计稿”或“TinyLlama 已完整交付”的表述。

## 2. 本机验证环境

验证日期：2026 年 3 月 11 日

- GPU：`MetaX C500`
- `mx-smi`：`2.2.9`
- `MACA`：`3.2.1.10`
- 驱动：`3.0.11`
- 编译器：`mxcc 1.0.0`
- Python：`3.10.10`
- PyTorch：`2.6.0+metax3.2.1.3`
- xmake：`v2.8.7+20240401`

## 3. 已完成验证

### 3.1 构建

```bash
XMAKE_ROOT=y xmake f --metax-gpu=y -cv
XMAKE_ROOT=y xmake -r
XMAKE_ROOT=y xmake install
```

### 3.2 CPU 基线

```bash
python test/test_tensor.py
python test/test_runtime.py --device cpu
python test/test_ops.py --device cpu
python test/test_infer.py --device cpu --test --model_id trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 --prompt hi --max_steps 1
```

### 3.3 MetaX 主链路

```bash
python test/test_runtime.py --device metax
python test/test_ops.py --device metax
python test/test_infer.py --device metax --test --model_id trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 --prompt hi --max_steps 1
```

## 4. 关键说明

- MetaX 不是 C++ 层的 CUDA drop-in 兼容平台，因此后端必须单独适配
- PyTorch 层保留 `torch.cuda` 语义，因此 Hugging Face 对照测试仍可继续复用 CUDA 命名空间
- 当前推理验证聚焦 `Qwen2`
- 当前机器没有 NVIDIA 硬件，所以没有新增 `--device nvidia` 的实机回归数据
- 根目录外部 PDF 只阅读，不纳入仓库提交

## 5. 提交材料入口

- 实现报告：[`report_zh.md`](report_zh.md)
- 复现流程：[`reproduce_zh.md`](reproduce_zh.md)
- PR 文案：[`pr_zh.md`](pr_zh.md)

以上 3 份文档已经覆盖课程提交需要的核心内容：

- `report_zh.md`：实现说明与验证结论
- `reproduce_zh.md`：最短复现流程
- `pr_zh.md`：可直接提交的 GitHub PR 标题与正文
