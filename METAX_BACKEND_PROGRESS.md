# MetaX 后端接入开发日志

最后更新：2026-03-04  
目标：先打通 MetaX 后端“接入路线”（设备枚举 + runtime + 编译开关 + Python 映射），随后再逐步迁移底层算子。

---

## 0. 约束与阶段目标

### 当前阶段（Route-up）
1. 新增 `metax` 设备类型并保持外部接口兼容。  
2. 可以在框架内识别 `metax`，并完成 runtime 层路由。  
3. 不在本阶段实现 MetaX 算子内核，算子执行失败属于预期。  

### 下一阶段（Operator Porting）
按优先级迁移：`linear -> rms_norm -> rope -> self_attention -> 其他算子`。

---

## 1. 里程碑记录

### M001 - MetaX 路线骨架接入
- 日期：2026-03-03
- 目标：接入 `metax` 设备路由与编译入口，不改现有 CPU/NVIDIA 行为。
- 改动文件：
  - `include/llaisys.h`
  - `src/device/runtime_api.hpp`
  - `src/device/runtime_api.cpp`
  - `src/device/metax/metax_runtime_api.maca`（新增）
  - `xmake.lua`
  - `xmake/metax.lua`（新增）
  - `python/llaisys/libllaisys/llaisys_types.py`
  - `test/test_utils.py`
  - `test/test_runtime.py`
  - `test/chat_server.py`
- 关键改动：
  1. 设备枚举新增 `LLAISYS_DEVICE_METAX`。  
  2. runtime 分发新增 `metax::getRuntimeAPI()`。  
  3. 新增 `--mx-gpu` 编译选项与 `ENABLE_METAX_API` 宏。  
  4. 新增 `src/device/metax` 运行时骨架（当前返回 `unsupported/no-device`）。  
  5. Python `DeviceType` 新增 `METAX`。  
  6. `test_utils` 新增 `metax` 的设备映射。  
  7. `test_runtime.py`、`chat_server.py` 的 CLI 支持 `--device metax`。  
- 状态：已完成（骨架接入）。
- 风险：
  - 目前尚未迁移 MetaX 算子后端，模型推理调用会落入 `unsupported`。
  - 需要 MetaX SDK/编译工具链信息后再落地 `src/ops/*/metax/*`。
- 验证记录：
  - `xmake f --mx-gpu=y -cv && xmake`：通过，且产出 `libllaisys-device-metax.a`。
  - `xmake install`：通过，已同步新 `libllaisys.so` 到 `python/llaisys/libllaisys/`。
  - `PYTHONPATH=python python test/test_runtime.py --device metax`：
    - 输出 `Found 0 metax devices`，按预期 `Skipped` 并 `Test passed`。
  - 排障备注：
    - 若直接运行 `python test/test_runtime.py --device metax` 出现 `DeviceType.METAX` 缺失，通常是解释器加载了旧安装包；使用 `PYTHONPATH=python` 或重新安装 Python 包可解决。
    - 远端 C500 登录尝试：`ssh metaX` 当前返回 `Permission denied (password)`，需先补齐远端免密认证或提供可用登录凭据后再继续远端编译。
  - 远端服务器实测（用户提供）：
    - `xmake f --mx-gpu=y -cv`：配置通过（`mx-gpu=true`）。
    - `xmake`：编译通过，日志包含 `libllaisys-device-metax.a` 归档与 `libllaisys.so` 链接成功。
    - `xmake install`：安装通过，已复制动态库到 `python/llaisys/libllaisys/`。
    - 结论：MetaX Route-up 骨架在远端 C500 环境可成功构建。

### M002 - MetaX 动态运行时接入（进行中）
- 日期：2026-03-03
- 目标：让 `metax` runtime 具备真实设备/内存/流接口能力，不再只返回骨架占位行为。
- 改动文件：
  - `src/device/metax/metax_runtime_api.maca`
  - `src/device/metax/metax_resource.hpp`（新增）
  - `src/device/metax/metax_resource.maca`（新增）
  - `test/test_runtime.py`
  - `xmake.lua`
- 关键改动：
  1. `metax_runtime_api` 改为动态加载 cudart-like 运行时（`dlopen + dlsym`）。  
  2. 接入函数：`cudaGetDeviceCount / cudaSetDevice / stream / malloc / memcpy` 等。  
  3. 增加 `LLAISYS_METAX_DEBUG=1` 诊断日志，输出库加载路径、失败原因、device count。  
  4. 增加 `LLAISYS_METAX_CUDART=/path/to/libcudart.so` 强制指定运行时库路径。  
  5. 修复 `test_runtime.py` 的设备循环打印（`Testing device {i}` -> `Testing device 0`）。
  6. `xmake.lua` 增加 `mx-gpu` 场景下 `add_syslinks("dl")`，确保 `dlopen` 依赖显式链接。
  7. 按既有目录风格补齐 `metax_resource.hpp/.cpp`，与 `cpu/nvidia` 的 `resource + runtime_api` 结构保持一致。
  8. MetaX 运行时库候选路径扩展：支持 `MACA_HOME/MACA_ROOT/MXGPU_LLVM_HOME/MXCC_HOME`，并补充 `/opt/maca-3.x` 版本化目录路径探测。
  9. 增加兼容库名探测：`libruntime_cu.so / libmcruntime.so / libmxc-runtime64.so`，覆盖不同 MACA 镜像打包差异。
  10. 新增 driver API 回退路径：当 `cuda*` 运行时符号缺失时，自动尝试 `cu*`（含 `_v2` 变体）完成 device/stream/memcpy 基础能力，适配仅提供 `libmcruntime` 的环境。
  11. 新增 MetaX 官方 `mx*` Runtime API 分支（`mxDeviceGetCount/mxSetDevice/mxMalloc/mxMemcpy` 等）；加载优先级改为 `cudart -> mx -> driver`，并新增 `LLAISYS_METAX_RUNTIME` 环境变量用于显式指定运行时库路径。
- 本地验证（开发机）：
  - `xmake f --mx-gpu=y -cv && xmake && xmake install`：通过。  
  - `PYTHONPATH=python python test/test_runtime.py --device metax`：通过。  
  - `LLAISYS_METAX_DEBUG=1 ...` 日志显示当前开发机加载的是 `libcudart.so`（非 `/opt/maca` 路径）。
- 说明：
  - 当前实现仍是 “cudart 兼容层” 路线，尚未调用 MetaX 专有算子库。  
  - 若系统同时存在 NVIDIA CUDA 与 MACA，建议显式设置 `LLAISYS_METAX_CUDART`，避免误加载到非目标运行时。
  - 代码风格对齐：`metax_runtime_api.maca` 保持“runtime API 转发层”职责，动态加载与底层实现下沉到 `metax_resource.maca`，与项目内 `resource + runtime_api` 分层一致。

### M003 - C500 构建链路兼容修复（xmake 2.8.7）
- 日期：2026-03-03
- 背景：服务器环境为 `xmake v2.8.7`，`mxcc 1.0.0`，`mc_runtime.h` 位于 `/opt/maca-3.3.0/include/mcr/`。
- 典型问题与定位：
  1. `error: unknown source file: *.maca`
     - 原因：`xmake 2.8.7` 不支持将 `.maca` 直接作为 `add_files(..., {sourcekind="cxx"})` 输入（已用最小工程复现）。
  2. `error: cannot find known tool script for /opt/.../mxcc`
     - 原因：旧版 xmake 将绝对路径工具名按“tool script”解析，`set_toolset(..., "$(env MXCC)")` 不兼容。
  3. `fatal error: mc_runtime.h: No such file or directory`
     - 原因：头文件真实目录是 `include/mcr`，不是 `include` 根目录。
  4. 在 `/tmp/maca_probe` 目录执行 `xmake f --mx-gpu=y` 报 `Invalid option: --mx-gpu=y`
     - 原因：命令运行目录错误，非 `llaisys` 项目根目录。
- 解决方案（已落地）：
  1. 保留 `.maca` 为主源码，构建期在 `build/_gen/metax` 自动生成 `*_wrapper.cpp`，由 xmake 编译 wrapper。
     - 避免 `.maca` 直接输入 xmake 导致的识别失败，同时不污染源码目录。
  2. 移除 `mxcc` 自定义 toolchain 依赖，避免旧版 xmake tool script 解析问题。
  3. 在 `xmake/metax.lua` 增加 `add_includedirs(path.join(root, "include", "mcr"))`。
  4. 统一服务器执行路径：必须在 `~/llaisys` 下执行 `xmake` 与 `test` 命令。
- 关键改动文件：
  - `xmake/metax.lua`
  - `src/device/metax/metax_runtime_api.maca`
  - `src/device/metax/metax_resource.maca`
- 服务器验证结果（用户实测）：
  - `xmake f --mx-gpu=y -c -v && xmake -r && xmake install`：通过；
  - 编译日志可见 `build/_gen/metax/metax_*_wrapper.cpp`；
  - `PYTHONPATH=python python test/test_runtime.py --device metax`：
    - 输出 `Found 1 metax devices`
    - `Testing device 0... Passed`
    - `Test passed!`
- 结论：
  - MetaX 路由已在 C500 服务器完成“可构建 + 可枚举设备 + runtime memcpy 基础能力”打通。
  - 后续进入算子迁移阶段（`src/ops/*/metax/*`）。

### M004 - 首个 MetaX 算子闭环（Add，已完成）
- 日期：2026-03-03
- 目标：按“一个算子一闭环”启动算子迁移，首个算子选 `add`，并在 C500 完成“编译-链接-加载-功能测试”全链路打通。
- 关键改动：
  1. 新增 `src/ops/add/metax/add_metax.hpp`、`src/ops/add/metax/add_metax.maca`。
  2. `add_metax.maca` 直接包含 `mc_runtime.h`，采用 kernel launch（`<<<grid, block>>>`）执行 `f32` elementwise add。
  3. `src/ops/add/op.cpp` 增加 `LLAISYS_DEVICE_METAX` 分发，直接进入 `metax::add`。
  4. `xmake/metax.lua` 新增 `llaisys-ops-metax` 目标，并在 `on_build` 中直接调用 `mxcc` 编译 `src/ops/*/metax/*.maca` 为 `.o`，再用 `ar` 打包成 `libllaisys-ops-metax.a`。
  5. `xmake.lua` 中 `llaisys-ops` 对 `llaisys-ops-metax` 增加依赖与链接传播，保证最终 `libllaisys.so` 能解析 metax 算子符号。
  6. `metax::add` 接口签名统一为 `void* / const void*`（声明/定义/调用一致），避免跨编译器 ABI 名字不一致导致的符号错配。
  7. `test/test_utils.py`：增加 metax 基线策略（torch 使用 CPU，拷贝方向按 `H2D/D2H/D2D` 自动切换）。
  8. `test/ops/add.py`：新增 `--device metax`；当前 metax 先验证 `f32`。
- 排障过程（关键问题 -> 原因 -> 解决）：
  1. 报错：`blockIdx/blockDim/threadIdx not declared`、`<<<>>>` 解析失败。
     - 原因：`.maca` 被 wrapper 方式交给 `gcc` 编译，而不是 `mxcc`。
     - 解决：算子侧不再用 wrapper，改为 `xmake/metax.lua` 手动调用 `mxcc -c add_metax.maca`。
  2. 报错：`Cuda SDK not found!`。
     - 原因：尝试走 xmake 的 `cu` 工具链路径，xmake 2.8.7 会强依赖 CUDA SDK 检测。
     - 解决：放弃 `cu` 路径，改用 `on_build` 直接执行 `mxcc`。
  3. 报错：`cannot find known tool script for mxcc`。
     - 原因：xmake 2.8.7 对 `mxcc` 作为工具链脚本识别不稳定。
     - 解决：不把 `mxcc` 注册为 xmake toolset，改为普通外部命令调用。
  4. 报错：`mxcc: language not recognized: 'MXMACA'`。
     - 原因：该版本 `mxcc` 不接受 `-x MXMACA`。
     - 解决：直接以 `.maca` 后缀输入编译，不再传 `-x MXMACA`。
  5. 报错：`undefined symbol: llaisys::ops::metax::add...`（Python `ctypes.CDLL` 加载失败）。
     - 原因：`libllaisys.so` 链接阶段未稳定拉入 `llaisys-ops-metax` 的目标符号，且早期存在函数签名不一致问题。
     - 解决：统一 `add` 签名为 `void*` 版本，并在 `llaisys-ops` 显式传播 `llaisys-ops-metax` 链接，最终链接顺序稳定后符号解析成功。
- 服务器最终验证（用户实测）：
  1. 构建通过：`xmake f --mx-gpu=y -c -v && xmake -r -v && xmake install`。
     - 日志可见：`mxcc ... -c src/ops/add/metax/add_metax.maca -o build/_gen/metax_ops_obj/add_add_metax.o`。
     - 日志可见：`ar -cr ... libllaisys-ops-metax.a ...add_add_metax.o`。
  2. 动态库符号确认：
     - `nm -D python/llaisys/libllaisys/libllaisys.so | c++filt | grep "llaisys::ops::metax::add"`
     - 输出：`T llaisys::ops::metax::add(void*, void const*, void const*, llaisysDataType_t, unsigned long)`。
  3. 功能测试通过：
     - `PYTHONPATH=python python test/ops/add.py --device metax`
     - 输出：`shape (2, 3)`、`shape (512, 4096)` 均通过，`Test passed!`。
- 状态：完成。

### M005 - MetaX Argmax 首版迁移（已完成）
- 日期：2026-03-04
- 目标：按 “MetaX 实现尽量对齐 CUDA 算子结构” 的原则，完成 `argmax` 的首版迁移并跑通 `cpu/nvidia/metax` 三平台测试入口。
- 关键改动：
  1. 新增 `src/ops/argmax/metax/argmax_metax.hpp`、`src/ops/argmax/metax/argmax_metax.maca`。  
  2. 在 `src/ops/argmax/op.cpp` 增加 `LLAISYS_DEVICE_METAX` 分发，路由到 `metax::argmax`。  
  3. `argmax_metax.maca` 对齐 CUDA 方案：线程级扫描 + warp 级规约 + warp leader 汇总。  
  4. warp 规约优先使用官方 API：`__shfl_down_sync(...)` + `warpSize`，并使用 `common/maca_fp16.h`、`common/maca_bfloat16.h` 官方类型/转换接口。  
  5. 数据类型支持：`f32/f16/bf16`；空张量行为与 NVIDIA 路径保持一致（`max_idx=0`，`max_val=0`）。  
  6. 索引类型保持 `int64_t`，对齐框架张量 dtype（`max_idx` 为 `i64`）。
- 当前实现状态：
  - kernel 配置为 `<<<1, 256>>>`（单 block 首版）；已具备 warp 级规约，后续可继续做多 block 两阶段归约。  
  - 功能测试已通过：`python test/ops/argmax.py --device metax`。
- 性能观察（用户服务器实测）：
  - 小规模（`shape=(4,)`）LLAISYS 已快于 Torch 基线。  
  - 中规模（`shape=(4096,)`）与 Torch 接近，仍有优化空间（主要在 launch 配置与并行度利用）。

### M006 - 三平台测试基线设备对齐修复（已完成）
- 日期：2026-03-04
- 目标：统一测试脚本在 `cpu/nvidia/metax` 三平台的 Torch 基线设备行为，避免对比口径不一致。
- 关键改动（`test/test_utils.py`）：
  1. `torch_baseline_device("metax")` 改为返回 `torch_device("metax")`（不再固定到 CPU）。  
  2. `torch_device("metax")` 映射为 `torch.device("cuda:{id}")`，匹配 mcPyTorch 的 CUDA 兼容暴露方式。  
  3. `torch_to_llaisys_memcpy_kind(...)` 与 `llaisys_to_torch_memcpy_kind(...)` 改为按源/目的张量实际驻留设备自动推导 `H2D/D2H/D2D`。
- 用户确认结果（服务器）：
  - `torch_baseline_device("metax") -> cuda:0`；`random_tensor(..., "metax")` 也在 `cuda:0`。  
  - `python test/ops/argmax.py --device metax --profile` 可稳定跑通并输出可比的 Torch/LLAISYS 时间。
- 结论：
  - 目前 `--device metax` 路径下，Torch 基线已按 MetaX 服务器上的 GPU 路径执行（非 CPU 基线）。

---

## 2. 验收口径（当前阶段）

### Route-up 验收
1. 编译：`xmake f --mx-gpu=y -cv && xmake` 可通过。  
2. 运行时：`python test/test_runtime.py --device metax` 可进入 MetaX runtime 路径（无设备时可跳过）。  
3. 不影响 CPU/NVIDIA 现有功能。  

### Operator-up 验收（当前仅 Add）
1. `src/ops/add/metax/add_metax.maca` 可由 `mxcc` 编译并归档进 `libllaisys-ops-metax.a`。  
2. `libllaisys.so` 中存在 `llaisys::ops::metax::add(...)` 导出符号。  
3. `PYTHONPATH=python python test/ops/add.py --device metax` 在 C500 可通过。  

---

## 3. 下一步计划

### M007（计划）- 迁移 `linear` MetaX 算子
1. 复用 M004 的 `.maca -> mxcc -> .o -> .a` 构建链路，新增 `src/ops/linear/metax/*`。  
2. 优先打通 `f32` correctness，再扩展 `f16/bf16`。  
3. 补齐 `test/ops/linear.py --device metax` 与性能 profile。  

### M008（计划）- Transformer 核心算子迁移
1. 迁移 `rms_norm -> rope -> self_attention -> swiglu`。  
2. 跑 `test/test_infer.py --test` 做端到端 correctness。  
3. 跑 `test/benchmark_infer.py` 与 torch/metax 基线做吞吐对比。

---

## 4. C500 排查手册（当前重点）

### 4.1 基本流程
1. `xmake f --mx-gpu=y -cv && xmake && xmake install`  
2. `PYTHONPATH=python LLAISYS_METAX_DEBUG=1 python test/test_runtime.py --device metax`

### 4.2 若仍显示 `Found 0 metax devices`
1. 检查可见运行时库：`ldconfig -p | rg libcudart`  
2. 查找 MACA 安装路径：`find /opt /usr/local -name 'libcudart.so*' 2>/dev/null`  
3. 显式指定库：  
   `export LLAISYS_METAX_CUDART=/opt/maca/tools/cu-bridge/lib64/libcudart.so`  
   `PYTHONPATH=python LLAISYS_METAX_DEBUG=1 python test/test_runtime.py --device metax`
4. 若仍为 0，保留调试日志并继续检查容器设备节点/cgroup 可见性（与 `mx-smi` 可见性不完全等价）。

### 4.3 若出现构建错误（xmake 2.8.7 兼容）
1. 错误 `unknown source file: *.maca`：确认已同步最新 `xmake/metax.lua`（应编译 `build/_gen/metax/*_wrapper.cpp`，而非直接编译 `.maca`）。
2. 错误 `cannot find known tool script for /opt/.../mxcc`：确认 `xmake/metax.lua` 中不再配置 `$(env MXCC)` 作为 `set_toolset`。
3. 错误 `Invalid option: --mx-gpu=y`：确认当前目录是 `~/llaisys`，不是临时 probe 目录。
4. 错误 `mc_runtime.h not found`：确认 `MACA_HOME` 已设置，且 `xmake/metax.lua` 包含 `include/mcr` 头路径。

---

## 5. 参考资料（官方）

- MetaX MACA Developer Guide（CUDA 兼容说明）：  
  https://repos.metax-tech.com/gitlab/maca/maca/-/wikis/Developer_Guide_cn/03_MACA_CUDA
- MetaX MACA Developer Guide（CUDA 项目迁移）：  
  https://repos.metax-tech.com/gitlab/maca/maca/-/wikis/Developer_Guide_cn/04_Migration_of_Existing_CUDA_Projects_to_MACA

本阶段用法说明（2026-03-03）：
1. 依据官方“cu-bridge”兼容路径，补充 `libcudart.so` 候选加载路径（如 `/opt/maca/tools/cu-bridge/lib64`）。  
2. 依据官方迁移建议，保持对 CUDA Runtime API 的兼容调用形态，降低后续从 NVIDIA 路线迁移算子的改造成本。  
3. 依据官方安装文档中的环境变量示例，补充对 `MACA_HOME`/`LD_LIBRARY_PATH`/`CUDA_PATH(CUCC_PATH)` 场景的兼容配置建议。  

---

## 6. 开发约定（2026-03-04）

1. MetaX 算子实现优先与 CUDA 算子“实现思路 + 代码结构”对齐，尽量做到替换官方接口即可迁移。  
2. MetaX 侧优先使用官方 API（不限于类型转换，包含 shuffle/warp 等）；确认无官方接口时再引入自定义实现。  
3. 环境约束：本地开发机仅有 RTX 4060；MetaX 显卡在远程服务器。涉及 MetaX 实机验证时，记录并提供可直接执行的命令。  
