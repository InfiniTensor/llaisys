# PR 文案（可直接提交）

## 标题

`feat: 落地 LLAISYS 项目 2 第二平台 MetaX/MACA 并补齐提交文档`

## 正文

本 PR 基于 `checkpoint/nvidia-done`，把项目 2 的第二平台从设计稿推进到 MetaX/MACA 的可测试实现状态，同时补齐了可直接提交作业的中文文档。

### 变更摘要

- 新增独立 `METAX` 设备类型，保留原有 `CPU` 和 `NVIDIA` 分支
- 新增 `--metax-gpu=y` 构建开关和 MetaX 专用 xmake 规则
- 使用 `mxcc`、`mc_runtime`、`mcblas` 打通第二平台 runtime
- 完成 `add`、`embedding`、`linear`、`rms_norm`、`rope`、`swiglu` 的 MetaX 路径
- `linear` 对接 `mcblasGemmEx`
- `argmax` 和 `self_attention` 先以 host fallback 保证链路正确
- Python 侧补齐 `DeviceType.METAX` 与测试入口
- 重写报告、复现、平台说明、面试材料和 README 提交入口

### 兼容性判断

- MetaX 不是 C/C++ 层的 CUDA drop-in 兼容平台，不能直接复用 `<cuda_runtime.h>` / `cuBLAS`
- 但 PyTorch 层提供了 CUDA 语义兼容，因此测试侧仍复用 `torch.cuda`
- 这也是本次实现采用“独立 MetaX 后端 + 复用 HF CUDA 对照”的原因

### 已验证命令

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

### 说明

- 当前机器是沐曦 MetaX C500，已确认 `MACA 3.2.1.10`、驱动 `3.0.11`、`mxcc 1.0.0`
- 当前机器没有 NVIDIA 硬件，因此这次没有新增 `--device nvidia` 的实机回归数据
- 本次推理验证聚焦 `Qwen2`，不再沿用旧文档里关于 `TinyLlama/Llama` 的完成声明
- 根目录外部 PDF 保持未跟踪状态，不提交进仓库

### 文档入口

- 提交总览：[`submission_zh.md`](submission_zh.md)
- 实现报告：[`report_zh.md`](report_zh.md)
- 复现流程：[`reproduce_zh.md`](reproduce_zh.md)
- MetaX 实现说明：[`metax_design_zh.md`](metax_design_zh.md)
- 面试问答：[`interview_qa_zh.md`](interview_qa_zh.md)
- 5 分钟讲稿：[`interview_script_5min_zh.md`](interview_script_5min_zh.md)
