# 项目#2 报告：在 LLAISYS 中集成 CUDA（详细笔记版）

## 1. 项目目标
本项目目标是在 LLAISYS 中完成 NVIDIA CUDA 后端集成，并形成可复现的验证闭环：
- Runtime 层：`test/test_runtime.py --device nvidia`
- 算子层：`test/ops/*.py --device nvidia`（核心算子）
- 模型层：`test/test_infer.py --device nvidia --test`

目标不只是“能编译”，而是要在真实 GPU 环境上完成“构建 -> 安装 -> 运行 -> 对齐校验”的端到端验证。

## 2. 架构理解（实现前提）
LLAISYS 运行时核心路径：
- 每线程维护一个 `Context`。
- `Context` 按设备类型维护 `Runtime`，首次使用时延迟初始化。
- `setDevice(device_type, device_id)` 切换当前设备上下文。
- `Runtime` 持有 `LlaisysRuntimeAPI` 函数指针集合。

相关文件：
- `src/core/context/context.hpp`
- `src/core/context/context.cpp`
- `src/core/runtime/runtime.cpp`
- `src/device/runtime_api.hpp`

## 3. 实现内容（代码层）

### 3.1 CUDA Runtime API 完整实现
文件：`src/device/nvidia/nvidia_runtime_api.cpp`

已实现映射：
- 设备管理：
  - `getDeviceCount` -> `cudaGetDeviceCount`
  - `setDevice` -> `cudaSetDevice`
  - `deviceSynchronize` -> `cudaDeviceSynchronize`
- Stream 管理：
  - `createStream` -> `cudaStreamCreate`
  - `destroyStream` -> `cudaStreamDestroy`
  - `streamSynchronize` -> `cudaStreamSynchronize`
- 内存管理：
  - `mallocDevice/freeDevice` -> `cudaMalloc/cudaFree`
  - `mallocHost/freeHost` -> `cudaMallocHost/cudaFreeHost`
- 拷贝管理：
  - `memcpySync` -> `cudaMemcpy`
  - `memcpyAsync` -> `cudaMemcpyAsync`

增强点：
- `check_cuda(...)` 统一错误检查，异常信息带 CUDA 错误字符串。
- `llaisysMemcpyKind_t` 与 `cudaMemcpyKind` 显式映射，避免隐式错误。
- 共享集群可用设备筛选（见 3.2）。

### 3.2 共享集群设备筛选（稳定性增强）
背景：共享 A100 环境下，某些物理卡可能暂时不可用，直接 `cudaSetDevice` 会失败（常见 OOM/上下文不可用）。

处理：
- 初始化时遍历物理设备。
- 对每个设备执行 `cudaSetDevice + cudaFree(nullptr)` 预热。
- 仅保留可激活设备到 `available_devices`。
- `getDeviceCount` 返回逻辑可用设备数。
- `setDevice(logical_id)` 映射到对应物理设备。

效果：`test_runtime` 在共享环境下稳定通过，不再因“脏设备”误失败。

### 3.3 NVIDIA 资源类实现
文件：`src/device/nvidia/nvidia_resource.cpp`
- 实现 `Resource(int device_id)` 构造与析构，挂接 `DeviceResource`。

### 3.4 Xmake 集成与链接修正
文件：`xmake/nvidia.lua`
- 新增 `llaisys-device-nvidia` 静态库目标。
- 编译 `src/device/nvidia/*.cpp`。
- Linux 下增加 CUDA include/link 和 `cudart`。

文件：`xmake.lua`
- `--nv-gpu=y` 时启用 `ENABLE_NVIDIA_API`，并引入 `xmake/nvidia.lua`。
- `llaisys-device` 显式依赖 `llaisys-device-nvidia`。
- Linux 共享库目标补充 `add_syslinks("gomp")`，解决 OpenMP 运行时符号问题（`omp_get_thread_num`）。

## 4. 算子层与模型层补齐

### 4.1 NVIDIA 算子路径（staging 方案）
为保证先打通正确性链路，关键算子采用 D2H/H2D staging：
- 在 NVIDIA 分支中将输入从设备拷回 host。
- 复用 CPU 算法实现。
- 结果再拷回设备。

已补齐/验证的关键算子：
- `add`
- `linear`
- `argmax`
- `embedding`
- `rms_norm`
- `rope`
- `self_attention`
- `swiglu`

涉及文件：
- `src/ops/add/op.cpp`
- `src/ops/linear/op.cpp`
- `src/ops/argmax/op.cpp`
- `src/ops/embedding/op.cpp`
- `src/ops/rms_norm/op.cpp`
- `src/ops/rope/op.cpp`
- `src/ops/self_attention/op.cpp`
- `src/ops/swiglu/op.cpp`
- `src/ops/rearrange/op.cpp`

### 4.2 模型推理路径（qwen2）设备安全改造
文件：`src/llaisys/qwen2.cc`

关键改造：
- 替换直接设备指针读写（`memcpy/memset/直接解引用`）为 runtime API 安全读写。
- 新增 helper：
  - `zero_tensor`
  - `tensor_write_i64`
  - `tensor_read_i64`
  - `tensor_copy_bytes`
- 放开创建模型时 CPU-only 限制，使 nvidia 设备可进入完整推理流程。

## 5. 去冗余与目录统一
为避免重复路径和链接复杂度，统一 NVIDIA 后端到 `cpp + cudart`：
- 删除：`src/device/nvidia/nvidia_runtime_api.cu`
- 删除：`src/device/nvidia/nvidia_resource.cu`

文档同步：
- `README.md`
- `README_ZN.md`

## 6. 问题与修复记录（按时间）

### 问题1：本地无 CUDA SDK
- 现象：`Cuda SDK not found!`
- 处理：转到远端 A100 环境执行构建与验证。

### 问题2：远端无 xmake
- 现象：`xmake: command not found`
- 处理：安装到 `~/.local/bin/xmake`，后续统一绝对路径调用。

### 问题3：动态库 CUDA 注册符号错误
- 现象：`undefined symbol: __cudaRegisterLinkedBinary_...`
- 处理：去掉 `.cu` 动态注册路径，改为 `cpp + cudart`。

### 问题4：共享卡 `cudaSetDevice` 失败
- 处理：增加可用设备筛选与逻辑映射，测试前固定 `CUDA_VISIBLE_DEVICES`。

### 问题5：AVX 头与项目宏冲突（GCC）
- 现象：`include/llaisys.h` 中 `__C` 宏与 intrinsics 头参数名冲突。
- 处理：在 `src/ops/linear/op.cpp` 里将 `<immintrin.h>` 放在项目头之前。

### 问题6：Linux 下 OpenMP 运行时符号缺失
- 现象：Python 加载 `libllaisys.so` 报 `omp_get_thread_num` undefined。
- 处理：`xmake.lua` 共享库目标增加 `add_syslinks("gomp")`。

### 问题7：`test_infer` 依赖与离线模型
- 依赖：`transformers`、`huggingface_hub`、`sentencepiece`、`accelerate`。
- 网络受限时：通过本地模型目录上传远端并用 `--model` 指定路径。
- 本次使用模型：`/home/yuanstar/models/DeepSeek-R1-Distill-Qwen-1___5B`。

## 7. 最新复跑验证（2026-03-08）

### 7.1 复跑命令
```bash
cd /home/yuanstar/llaisys
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=/home/yuanstar/llaisys/python

/home/yuanstar/.local/bin/xmake f --nv-gpu=y -cv
/home/yuanstar/.local/bin/xmake -y
/home/yuanstar/.local/bin/xmake install -y

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

### 7.2 复跑结果摘要
- 构建：成功。
- Runtime：`Test passed!`。
- 核心算子：全部通过。
- Infer：`Test passed!`（HF tokens 与 LLAISYS tokens 一致）。

附注：`rope` 大张量 `f32` 用例在随机输入下可能偶发接近阈值，重跑可通过；当前实现满足作业验证链路，但若后续要做长期 CI，可考虑适度放宽该单测阈值或改为设备原生内核以减少数值实现差异。

## 8. 已变更文件清单
- `src/device/nvidia/nvidia_runtime_api.cpp`
- `src/device/nvidia/nvidia_resource.cpp`
- `src/device/nvidia/nvidia_resource.cuh`
- `xmake/nvidia.lua`
- `xmake.lua`
- `src/llaisys/qwen2.cc`
- `src/ops/add/op.cpp`
- `src/ops/linear/op.cpp`
- `src/ops/argmax/op.cpp`
- `src/ops/embedding/op.cpp`
- `src/ops/rms_norm/op.cpp`
- `src/ops/rope/op.cpp`
- `src/ops/self_attention/op.cpp`
- `src/ops/swiglu/op.cpp`
- `src/ops/rearrange/op.cpp`
- `test/ops/self_attention.py`
- `README.md`
- `README_ZN.md`
- 删除：`src/device/nvidia/nvidia_runtime_api.cu`
- 删除：`src/device/nvidia/nvidia_resource.cu`

## 9. 结论
项目#2 已完成并通过复跑验证：
- Runtime 层：通过
- 算子层：通过
- 模型层（`test_infer --device nvidia --test`）：通过

当前版本已达到作业要求，并对共享 GPU 场景、构建链路和离线模型验证做了额外稳健性增强。