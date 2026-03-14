# LLAISYS 作业提交总览

## 1. 当前提交范围

- 项目 2 第二平台：MetaX/MACA
- 目标状态：从“设计稿”推进到“真实机器上可测试”
- 约束：不改坏现有 `CPU + NVIDIA` 路径

## 2. 本机验证环境

- GPU：`MetaX C500`
- `mx-smi`：`2.2.9`
- `MACA`：`3.2.1.10`
- 驱动：`3.0.11`
- 编译器：`mxcc 1.0.0`
- Python：`3.10.10`
- PyTorch：`2.6.0+metax3.2.1.3`
- xmake：`v2.8.7+20240401`

## 3. 当前机器已实跑验证

### 3.1 构建

```bash
XMAKE_ROOT=y xmake f --metax-gpu=y -cv
XMAKE_ROOT=y xmake -r
XMAKE_ROOT=y xmake install
```

### 3.2 MetaX 主链路

```bash
python test/test_runtime.py --device metax
python test/test_ops.py --device metax
python test/test_infer.py --device metax --test --model_id trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 --prompt hi --max_steps 1
```

## 4. 关键说明

- 为保持 PR 干净，本次提交只包含实现代码与正式提交文档；本地学习材料与外部 PDF 均未纳入仓库
- MetaX 不是 C++ 层的 CUDA drop-in 兼容平台，因此后端必须单独适配
- PyTorch 层保留 `torch.cuda` 语义，因此 Hugging Face 对照测试仍可继续复用 CUDA 命名空间
- 当前推理验证聚焦 `Qwen2`
- CPU 基线不在当前沐曦机器上重复列为实跑结果
- 当前机器没有 NVIDIA 硬件，所以没有新增 `--device nvidia` 的实机回归数据
- 根目录外部 PDF 只阅读，不纳入仓库提交

## 5. 提交材料入口

- 实现报告：[`report_zh.md`](report_zh.md)
- 复现流程：[`reproduce_zh.md`](reproduce_zh.md)
- PR 文案：[`pr_zh.md`](pr_zh.md)

以上 3 份文档配合当前代码改动与实际 GitHub PR，可覆盖课程提交需要的核心内容：

- `report_zh.md`：实现说明与验证结论
- `reproduce_zh.md`：最短复现流程
- `pr_zh.md`：可直接提交的 GitHub PR 标题与正文
