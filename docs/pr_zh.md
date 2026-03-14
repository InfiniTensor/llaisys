# GitHub PR 提交模板

## 标题

`feat: support MetaX/MACA as the second backend for project 2`

## 正文

本 PR 完成 LLAISYS 项目 2 第二平台 `MetaX/MACA` 的实际接入与验证，并补齐课程提交所需中文文档。

### 主要改动

- 新增独立 `METAX` 设备类型，保留原有 `CPU` 和 `NVIDIA` 路径
- 新增 `--metax-gpu=y` 构建开关和 MetaX 专用 xmake 规则
- 使用 `mxcc`、`mc_runtime`、`mcblas` 打通第二平台 runtime
- 完成 `add`、`embedding`、`linear`、`rms_norm`、`rope`、`swiglu` 的 MetaX 路径
- `linear` 对接 `mcblasGemmEx`
- `argmax` 和 `self_attention` 先以 host fallback 保证主链路正确
- Python 侧补齐 `DeviceType.METAX` 与测试入口
- 补齐提交总览、实现报告与复现流程
- 本 PR 只包含实现代码与正式提交文档，本地学习材料、外部 PDF 与 handoff 文档未纳入提交

### 验证结果

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

- MetaX 不是 C/C++ 层的 CUDA drop-in 兼容平台，因此采用独立后端适配
- 当前机器是沐曦 `MetaX C500`，已确认 `MACA 3.2.1.10`、驱动 `3.0.11`、`mxcc 1.0.0`
- 当前机器没有 NVIDIA 硬件，因此本次没有新增 `--device nvidia` 的实机回归数据
- 当前推理验证聚焦 `Qwen2`
- 根目录外部 PDF 保持未跟踪状态，不提交进仓库

### 提交文档

- 提交总览：[`submission_zh.md`](submission_zh.md)
- 实现报告：[`report_zh.md`](report_zh.md)
- 复现流程：[`reproduce_zh.md`](reproduce_zh.md)
