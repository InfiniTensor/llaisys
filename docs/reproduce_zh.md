# LLAISYS 项目 2 第二平台复现流程（MetaX/MACA）

## 1. 适用范围

本文档对应当前仓库的可提交状态，目标是复现：

- CPU 基线路径
- MetaX/MACA 第二平台路径
- `runtime -> ops -> infer` 的完整验证链路

如果你当前机器不是沐曦平台，而是受限沙箱、纯 CPU 或纯 NVIDIA 机器，本文中的 MetaX 测试无法直接复现。

## 2. 环境检查

在仓库根目录先执行下面几组命令，确认平台和工具链是对的：

```bash
mx-smi
mxcc --version
python --version
python -c "import torch; print(torch.__version__); ok = torch.cuda.is_available(); print(ok); print(torch.cuda.get_device_name(0) if ok else 'no visible metax device')"
echo "$LD_LIBRARY_PATH"
ls /opt/maca/include | head
ls /opt/maca/lib | head
ls /opt/mxdriver/lib | head
```

参考环境：

- `mx-smi 2.2.9`
- `MetaX C500`
- `MACA 3.2.1.10`
- 驱动 `3.0.11`
- `mxcc version 1.0.0`
- Python `3.10.10`
- PyTorch `2.6.0+metax3.2.1.3`

如果 `LD_LIBRARY_PATH` 里没有 MetaX 运行库，建议补上：

```bash
export LD_LIBRARY_PATH=/opt/maca/lib:/opt/mxdriver/lib:$LD_LIBRARY_PATH
```

## 3. 依赖准备

### 3.1 xmake

当前仓库用 `xmake` 构建。如果系统里没有，可以安装：

```bash
apt-get update
apt-get install -y xmake
```

如果系统源里没有 `xmake` 包，需要改用 xmake 官方安装方式；只要最终 `xmake --version` 可用即可。

注意：如果你以 root 身份运行，后续所有 `xmake` 命令都要额外带上 `XMAKE_ROOT=y`。

### 3.2 Python 依赖

当前仓库在 `test/test_infer.py` 的 Hugging Face 路径上至少需要以下 Python 依赖：

```bash
python -m pip install transformers huggingface_hub accelerate
```

如果当前平台已经预装对应版本，可以跳过这一步。

## 4. 构建

### 4.1 CPU 默认构建

```bash
XMAKE_ROOT=y xmake -r
```

### 4.2 MetaX 构建

```bash
XMAKE_ROOT=y xmake f --metax-gpu=y -cv
XMAKE_ROOT=y xmake -r
XMAKE_ROOT=y xmake install
```

说明：

- `--metax-gpu=y` 会打开 MetaX 编译路径
- `.cu` 文件由 `mxcc` 负责编译
- 安装步骤会把 Python 侧需要的共享库放到可加载位置

## 5. 测试顺序

### 5.1 CPU 基线

```bash
python test/test_tensor.py
python test/test_runtime.py --device cpu
python test/test_ops.py --device cpu
python test/test_infer.py --device cpu --test --model_id trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 --prompt hi --max_steps 1
```

### 5.2 MetaX runtime

```bash
python test/test_runtime.py --device metax
```

预期结果：

- 设备数大于 0
- host/device 内存分配正常
- H2D、D2H、D2D 拷贝通过

### 5.3 MetaX ops

```bash
python test/test_ops.py --device metax
```

当前实现说明：

- `linear` 走 `mcblasGemmEx`
- `add`、`embedding`、`rms_norm`、`rope`、`swiglu` 走 MetaX kernel
- `argmax` 与 `self_attention` 当前是 host fallback

### 5.4 MetaX infer

如果本地没有现成模型目录，直接用公开的小模型：

```bash
python test/test_infer.py --device metax --test --model_id trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 --prompt hi --max_steps 1
```

如果你已经准备好本地 Qwen2 模型目录，也可以这样跑：

```bash
python test/test_infer.py --device metax --test --model /path/to/local/qwen2_model --prompt hi --max_steps 1
```

预期结果：

- Hugging Face 和 LLAISYS 的 token 序列严格一致
- 输出末尾打印 `Test passed!`

## 6. 常见问题

### 6.1 `xmake` 提示 root 用户危险，直接退出

补上环境变量再执行：

```bash
XMAKE_ROOT=y
```

例如：

```bash
XMAKE_ROOT=y xmake f --metax-gpu=y -cv
```

### 6.2 MetaX 设备在受限环境里不可见

MetaX 相关测试必须跑在真实沐曦机器上。  
如果当前环境对设备节点、驱动或运行库做了限制，`mcGetDeviceCount` 可能会失败，这不是仓库逻辑错误。

### 6.3 为什么测试里 `metax` 映射成了 `torch.cuda`

因为本机的 MetaX PyTorch 暴露的是 CUDA 语义接口，而不是新的 `torch.device("metax")` 命名空间。  
所以 Hugging Face 对照仍然走 `torch.cuda`，但 LLAISYS 自己的设备类型是独立 `METAX`。
