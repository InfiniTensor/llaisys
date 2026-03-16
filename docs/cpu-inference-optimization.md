# LLAISYS CPU 推理优化文档

本文档记录针对项目 #1「优化 LLAISYS 的 CPU 推理」所做的改动，主要包括：**OpenMP 多线程并行**、**AVX2 SIMD 向量化**（FP32 linear），以及为兼容系统头文件所做的 **LLAISYS_EXTERN_C 宏重命名**。

---

## 一、背景与目标

### 1.1 问题

- 未优化前，LLAISYS 的模型推理速度相比 PyTorch 明显更慢。
- 主要瓶颈在 **linear**（矩阵乘法）算子，该算子在 Transformer 中调用最频繁、耗时占比最高。
- 原始实现为三重循环的朴素矩阵乘，未利用多核与 SIMD。

### 1.2 优化思路（与 README/项目说明一致）

| 方法           | 说明 |
|----------------|------|
| **SIMD**       | 使用 AVX2/AVX-512 等指令一次处理多个 float，提高吞吐。 |
| **OpenMP**     | 用多线程并行化算子，使 linear 等能利用多核 CPU。 |
| **第三方库**   | 可选用 OpenBLAS、MKL、Eigen 等（本次未接入，见后续可选）。 |

本次实现采用 **OpenMP + AVX2**，在不引入额外依赖的前提下提升 CPU 推理速度。

---

## 二、优化一：OpenMP 多线程

### 2.1 思路

- linear 计算 `Y = X W^T + b`，其中 `out` 形状为 `(B, M)`，`in` 为 `(B, K)`，`weight` 为 `(M, K)`。
- 外层按行（B 维）并行：每个线程负责若干行输出，互不写同一位置，无需加锁。
- 使用 `schedule(static)` 静态划分，便于缓存局部性。

### 2.2 修改内容

**xmake.lua**

- 在 `llaisys-ops` 目标中增加：
  - `add_cxflags("-fopenmp")`、`add_mxflags("-fopenmp")`、`add_ldflags("-fopenmp")`。
- 在最终动态库目标 `llaisys` 中增加：
  - `add_ldflags("-fopenmp")`，以便链接 OpenMP 运行时。

**src/ops/linear/op.cpp**

- 在文件顶部增加（可选）：`#ifdef _OPENMP` 时 `#include <omp.h>`。
- 在模板函数 `linear_impl` 的外层循环（B 维）前增加：

```cpp
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
for (size_t i = 0; i < B; i++) {
    // ...
}
```

- 对所有 dtype（F32 / F16 / BF16）均生效；FP32 在启用 AVX2 时走 `linear_f32_avx2`，其内部同样使用上述同一套 OpenMP 并行。

### 2.3 效果

- 多核机器上，B 或 M 较大时能明显利用多核，线性层耗时随核数近似线性下降（受内存带宽限制会有所折扣）。

---

## 三、优化二：AVX2 + FMA（仅 FP32 linear）

### 3.1 思路

- 内层 K 维是连续内存上的点积，适合用 SIMD 一次处理多个 float。
- AVX2 提供 256 位寄存器，一次处理 **8 个 float**；FMA（Fused Multiply-Add）一条指令完成 `a*b+c`，减少舍入与指令数。
- 仅对 **FP32** 实现 AVX2 路径；F16/BF16 仍走原有 `linear_impl`（内部用 float 累加），避免重复实现半精度 SIMD。

### 3.2 修改内容

**xmake.lua**

- 在 `llaisys-ops` 中，当 `is_arch("x86_64")` 时增加：
  - `add_cxflags("-mavx2", "-mfma")`，使编译器生成 AVX2/FMA 指令并定义 `__AVX2__`。

**src/ops/linear/op.cpp**

1. **头文件顺序**  
   - 在包含任何项目头文件之前，先写：
   - `#ifdef __AVX2__`  
   - `#include <immintrin.h>`  
   - `#endif`  
   - 避免项目中的宏（见第四节）与 `<immintrin.h>` 及其间接包含的系统头文件中的符号冲突。

2. **AVX2 版 FP32 实现**  
   - `linear_f32_avx2(out, in, weight, bias, B, M, K)`：
     - 外层 B 维用 OpenMP 并行（与 `linear_impl` 一致）。
     - 对每个 `(i, j)`，内层 K 维：
       - 用 `_mm256_loadu_ps` 每次读 8 个 float；
       - 用 `_mm256_fmadd_ps(a, b, sum8)` 做乘加；
       - K 不是 8 的倍数时，剩余标量补齐。
     - 对 8 路累加结果做水平求和：`hsum_avx(sum8)`，得到标量 `sum`，再加 bias 写入 `out[i*M+j]`。
   - `hsum_avx(__m256 v)`：将 256 位寄存器中 8 个 float 相加为 1 个 float（用 `_mm256_castps256_ps128`、`_mm256_extractf128_ps`、`_mm_add_ps`、`_mm_movehdup_ps`、`_mm_movehl_ps`、`_mm_add_ss` 等实现）。

3. **分支选择**  
   - 在 `linear_cpu` 的 `LLAISYS_DTYPE_F32` 分支中：
     - 若定义了 `__AVX2__`，则调用 `linear_f32_avx2(...)`；
     - 否则调用原有 `linear_impl<float>(...)`。
   - F16/BF16 仍只走 `linear_impl`。

### 3.3 平台说明

- **x86_64**：默认开启 `-mavx2 -mfma`，FP32 linear 使用 AVX2 路径。
- **ARM / 其他架构**：不添加上述编译选项，FP32 仍为 OpenMP + 标量三重循环；后续可仿照实现 NEON 等 SIMD 版本。

---

## 四、兼容性修复：LLAISYS_EXTERN_C 宏

### 4.1 问题

- 项目在 `include/llaisys.h` 中用宏 `__C` 表示 `extern "C"`（C++ 时）或空（C 时）。
- 系统头文件（如 GCC 的 `<immintrin.h>` 间接包含的 `<ia32intrin.h>`）中，`__C` 被用作**参数名**。
- 开启 AVX2 并包含 `<immintrin.h>` 后，这些系统头在展开时会把参数名 `__C` 错误替换成 `extern "C"`，导致编译错误。

### 4.2 修改内容

- 在 **include/llaisys.h** 中：
  - 将 `#define __C extern "C"` 改为 `#define LLAISYS_EXTERN_C extern "C"`；
  - 将 `#define __C` 改为 `#define LLAISYS_EXTERN_C`。
- 在所有使用 `__C` 的地方改为 **LLAISYS_EXTERN_C**，涉及文件包括：
  - **头文件**：`include/llaisys.h`、`include/llaisys/models/qwen2.h`、`include/llaisys/runtime.h`、`include/llaisys/tensor.h`、`include/llaisys/ops.h`；
  - **实现**：`src/llaisys/llaisys_tensor.hpp`、`src/llaisys/runtime.cc`、`src/llaisys/tensor.cc`、`src/llaisys/ops.cc`、`src/llaisys/qwen2.cc`（两处 `__C {` 均改为 `LLAISYS_EXTERN_C {`）。

这样在任意源文件中先包含 `<immintrin.h>` 再包含项目头，也不会再与系统头中的 `__C` 冲突。

---

## 五、构建与验证

### 5.1 构建

```bash
cd /path/to/llaisys
xmake build llaisys
xmake install llaisys
```

- `xmake install` 会将生成的 `libllaisys.so`（或 Windows 下 `llaisys.dll`）复制到 `python/llaisys_py/libllaisys/`，供 Python 调用。

### 5.2 正确性测试

```bash
export PYTHONPATH="/path/to/llaisys/python:$PYTHONPATH"
python test/ops/linear.py --device cpu
```

- 应通过所有 shape 与 dtype 的测试（含 (512, 4096) 等大矩阵）。

### 5.3 性能对比（profile）

```bash
export PYTHONPATH="/path/to/llaisys/python:$PYTHONPATH"
python test/ops/linear.py --device cpu --profile
```

- 脚本会对 PyTorch 与 LLAISYS 的 linear 做 warmup + 多次重复计时，并打印两者耗时（ms）。
- 大矩阵（如 512×4096 × 4096×4096）下，预期 LLAISYS 相对未优化版本有明显加速（具体倍数与 CPU 核数、是否支持 AVX2 有关）。

### 5.4 实际推理体感

```bash
.venv/bin/python -m llaisys_py.server --model /path/to/DeepSeek-R1-Distill-Qwen-1___5B --port 8002
```

- 与优化前对比：首 token 延迟与后续 token 延迟应有所下降，尤其在多核、支持 AVX2 的 x86_64 上。

### 5.5 性能分析：优化前后对比

要量化「OpenMP + AVX2」带来的提升，可采用下面三种方式（由简到繁）。

#### 方法一：用 OMP_NUM_THREADS 看多线程收益（无需改代码、无需两套构建）

在同一台机器、同一套已开启 OpenMP 的构建下，只改变线程数对比耗时：

```bash
export PYTHONPATH="/path/to/llaisys/python:$PYTHONPATH"

# 单线程（相当于“无多线程优化”的耗时）
OMP_NUM_THREADS=1 python test/ops/linear_bench.py --device cpu --dtype f32

# 多线程（例如 8 核）
OMP_NUM_THREADS=8 python test/ops/linear_bench.py --device cpu --dtype f32
```

保存两次输出的 `lla_ms`，则 **多线程加速比 ≈ 单线程时间 / 多线程时间**。例如单线程 200 ms、8 线程 35 ms，加速比约 5.7x。

#### 方法二：固定 benchmark 脚本，保存“优化前 / 优化后”数据

使用 `test/ops/linear_bench.py` 做**可复现**的计时，输出便于 diff 或写脚本解析：

```bash
# 优化前：例如先 checkout 到未加 OpenMP/AVX2 的提交，构建并安装后
xmake build && xmake install
python test/ops/linear_bench.py --device cpu --repeat 100 --json > baseline.json

# 优化后：切回当前代码，重新构建安装
xmake build && xmake install
python test/ops/linear_bench.py --device cpu --repeat 100 --json > optimized.json

# 对比（可用 jq 或手写脚本算 speedup = baseline_ms / optimized_ms）
```

同一台机器、同一 `--repeat` 下，直接比较各 shape/dtype 的 `lla_ms` 即可得到优化倍数。

#### 方法三：AVX2 开/关对比（需两套构建）

若想单独看 **AVX2 SIMD** 的收益（不含多线程差异），需要两次构建：

1. **无 AVX2 构建**：在 `xmake.lua` 的 `llaisys-ops` 中临时注释掉 `add_cxflags("-mavx2", "-mfma")`，然后 `xmake build && xmake install`，运行 `linear_bench.py` 保存结果（例如 `no_avx2.json`）。
2. **有 AVX2 构建**：恢复 `-mavx2 -mfma`，重新 `xmake build && xmake install`，再跑一次保存（例如 `with_avx2.json`）。

对比两者在 **FP32**、同一 shape 下的 `lla_ms`，即可得到 AVX2 带来的加速比。F16/BF16 当前无 AVX2 路径，对比意义不大。

#### 建议记录格式

- 每次 benchmark 注明：**机器（CPU 型号、核数）、OMP_NUM_THREADS、repeat、warmup**。
- 重点关注大矩阵：**out (512, 4096), x (512, 4096), w (4096, 4096)**，dtype **f32**，与 PyTorch 的耗时对比可作为参考（见 5.3）。

#### 如何报告性能提升

1. **生成对比报告**：用两份 JSON 跑报告脚本，直接得到表格和加速比。

```bash
# 单线程 vs 多线程
OMP_NUM_THREADS=1 python test/ops/linear_bench.py --device cpu --dtype f32 --json > single.json
OMP_NUM_THREADS=8 python test/ops/linear_bench.py --device cpu --dtype f32 --json > multi.json
python test/ops/linear_bench_report.py single.json multi.json
```

输出示例：
```
======================================================================
Linear 性能对比报告
======================================================================
  基准: single.json  (e.g. 优化前 / 单线程)
  对比: multi.json   (e.g. 优化后 / 多线程)

shape                         dtype   基准(ms)     对比(ms)     加速比
----------------------------------------------------------------------
[512, 4096] @ [4096, 4096]    f32        2537.22        82.02       30.94x
----------------------------------------------------------------------
说明: 加速比 = 基准耗时 / 对比耗时，>1 表示对比版本更快。
======================================================================
```

2. **书面报告建议结构**（可粘贴到 README / 实验报告）：
   - **环境**：CPU 型号、核数、OMP_NUM_THREADS、repeat/warmup。
   - **测试内容**：shape（如 512×4096 @ 4096×4096）、dtype（f32）。
   - **结果**：基准耗时（ms）、优化后耗时（ms）、**加速比**（基准/优化后）。
   - **结论**：例如「在 8 核机器上，OpenMP 多线程使 linear (f32) 大矩阵耗时从 xxx ms 降至 xxx ms，加速约 x.x 倍。」

---

## 六、涉及文件一览

| 文件 | 修改要点 |
|------|----------|
| **xmake.lua** | llaisys-ops：OpenMP 编译/链接选项；x86_64 下 -mavx2 -mfma。llaisys：-fopenmp 链接。 |
| **src/ops/linear/op.cpp** | 顶部条件包含 immintrin.h；linear_impl 外层 B 维 OpenMP；linear_f32_avx2 + hsum_avx；linear_cpu 中 F32 分支选 AVX2 或标量。 |
| **src/ops/self_attention/op.cpp** | 条件包含 omp.h；parallel 区内线程私有 scores，对 qlen 做 omp for；typed 写回对 total 做 omp for。 |
| **src/ops/rms_norm/op.cpp** | 条件包含 omp.h；rms_norm_impl 外层 rows 循环 omp parallel for。 |
| **src/ops/swiglu/op.cpp** | 条件包含 omp.h；swiglu_impl 外层 n 循环 omp parallel for。 |
| **src/ops/rope/op.cpp** | 条件包含 omp.h；rope_impl 外层 seq_len 循环 omp parallel for。 |
| **include/llaisys.h** | __C → LLAISYS_EXTERN_C。 |
| **include/llaisys/*.h**、**include/llaisys/models/qwen2.h** | 所有 __C 改为 LLAISYS_EXTERN_C。 |
| **src/llaisys/*.cc**、**src/llaisys/llaisys_tensor.hpp** | 所有 __C 改为 LLAISYS_EXTERN_C。 |

---

## 七、其他算子的 OpenMP 并行（已实现）

在 `self_attention`、`rms_norm`、`swiglu`、`rope` 的外层循环上已增加 OpenMP 并行，与 linear 共用同一套 `-fopenmp` 编译/链接选项，无需额外配置。

| 算子 | 并行维度 | 说明 |
|------|----------|------|
| **self_attention** | `qlen`（query 序列长度） | 每个线程私有 `scores` 缓冲区，`#pragma omp parallel` + `#pragma omp for`；F16/BF16 的 cast 写回用 `#pragma omp parallel for`。 |
| **rms_norm** | `rows`（行数） | 按行独立，`#pragma omp parallel for schedule(static)`。 |
| **swiglu** | 元素下标 `i`（总元素数 n） | 逐元素独立，`#pragma omp parallel for schedule(static)`。 |
| **rope** | `seq_len`（序列长度） | 每帧独立，`inv_freq` 只读共享；`#pragma omp parallel for schedule(static)`。 |

验证：`python test/ops/self_attention.py --device cpu`、`test/ops/rms_norm.py`、`test/ops/swiglu.py`、`test/ops/rope.py` 均已通过。

---

## 八、可选后续优化

1. **其他算子**  
   `add`、`embedding`、`rearrange`、`argmax`、`sample` 等若在 profile 中占比高，可同样对外层循环加 `#pragma omp parallel for`。

2. **BLAS 库**  
   将 FP32 linear 改为调用 OpenBLAS 的 `cblas_sgemm` 或 MKL 的等效接口，在 xmake 中增加对应依赖与链接，通常能获得更好性能，但需处理跨平台与依赖安装。

3. **ARM NEON**  
   在 ARM 架构下为 FP32 linear 实现 NEON 版本（一次处理 4 个 float），并在对应架构的编译选项中启用。

4. **BF16/F16 SIMD**  
   若模型以 BF16/F16 为主，可为半精度 linear 增加 AVX2/NEON 的 16 位或 32 位累加路径，以进一步提升半精度推理速度。

---

## 九、参考

- 项目 README / README_ZN 中「项目 #1：优化 LLAISYS 的 CPU 推理」说明。
- 算子性能分析：`python test/ops/linear.py --device cpu --profile`；优化前后对比：`python test/ops/linear_bench.py --device cpu [--dtype f32] [--json]`，见 5.5 节。
- OpenMP：<https://www.openmp.org/>
- Intel Intrinsics Guide（AVX2/FMA）：<https://www.intel.com/content/www/us/en/docs/intrinsics-guide/>

---

## 十、xmake 构建失败排查

若执行 `xmake build llaisys` 出现 `error:` 且无具体信息（或提示 `> in src/ops/argmax/op.cpp` 等），多为 xmake 将「add_cxflags("-fPIC") is ignored」等提示当作错误。当前已做处理：

- **xmake.lua**：全局 `set_policy("check.auto_ignore_flags", false)`；各 target 的 `add_cxflags("-fPIC", ...)` 已加 `{force = true}`；所有 target 已改为 `set_warnings("all")`（不再使用 `"error"`），避免警告被当成错误。
- **xmake/cpu.lua**：同上，且为 `llaisys-device-cpu`、`llaisys-ops-cpu` 设置了 `set_policy`。

**建议操作：**

1. 清理后重新构建：`rm -rf build && xmake build llaisys`（注意是 `rm` 不是 `m`）。  
   若报 `invalid argument: llaisys`，则只构建默认目标：`xmake build`（默认会构建 llaisys）。
2. 若构建仍报空 `error:`，多为 xmake 将「-fPIC is ignored」等提示当错误。当前已改为在根 xmake.lua **顶层** 只加一次 `add_cxflags("-fPIC", {force = true})`，各 target 内不再单独加 `-fPIC`，从而避免该检查。
3. 查看完整输出时，部分 xmake 要求选项在 `build` 之后：`xmake build llaisys -v 2>&1 | tee build.log`，再查看 `build.log` 中的 `warning:` / `error:`。
4. 若为「No space left on device」，需先清理磁盘再构建。

---

## 十一、流式对话 Prefill 逻辑 Bug 修复（服务端）

### 11.1 现象

使用 DeepSeek-R1-Distill-Qwen 等模型时，通过 FastAPI 流式接口（SSE）对话，模型回复出现**退化输出**：整段只出现 `1\n2\n3\n...` 或 `1 1 1` 等数字与换行，触发现有的 `_is_degenerate_output` 检测后提前停止，前端显示「（回复异常，请重试。）」。非流式 `generate()` 或直接用引擎脚本测试时，同一模型、同一 prompt 可正常生成中文。

### 11.2 原因

流式分支 `_stream_response` 中，当 `prefix_len == 0`（新会话、无 KV 缓存）时：

1. 先执行 `model.reset_kv_cache()`，然后进入 `for _ in range(n_remaining)` 循环。
2. **首步**调用 `model.next_token(...)` 时，传入的是：
   ```python
   tokens[-1:] if len(tokens) > 1 else tokens
   ```
   此时 `tokens = list(input_ids)`（例如 25 个 token），因此实际传入的是 **仅最后一个 token**（prompt 的结尾），而不是完整 prompt。

3. C++ 端 `llaisysQwen2ModelInfer` 收到的是「长度为 1 的序列」，相当于只做了一次 **decode 步**，没有对完整 prompt 做 **prefill**。KV cache 里只有 1 个位置，模型从未看到用户问题，后续自回归生成就退化成无意义的数字序列。

总结：**首步未做 prefill，只对 prompt 的最后一个 token 做了一次解码**，导致模型上下文错误、输出崩溃。

### 11.3 修改内容

**文件：`python/llaisys_py/server/app.py`**

在 `_stream_response` 的生成循环内，对**首步**区分处理：

- **首步且 `next_id is None` 且 `len(tokens) > 1`**（即 `prefix_len == 0` 的第一次调用）：传入**完整** `tokens`（即完整 `input_ids`），让 C++ 端对整段 prompt 做 prefill，并返回第一个生成 token。
- **其余步**：仍只传 `tokens[-1:]`，做单 token decode。

核心改动示例：

```python
# 首步且 prefix_len==0 时必须传入完整 prompt 做 prefill，否则只传最后一个 token 做 decode
if next_id is None and len(tokens) > 1:
    next_id = model.next_token(tokens, temperature=..., top_k=..., top_p=..., seed=...)
else:
    next_id = model.next_token(
        tokens[-1:] if len(tokens) > 1 else tokens,
        temperature=..., top_k=..., top_p=..., seed=...,
    )
```

### 11.4 验证

1. **最小复现脚本**（不经过 FastAPI，直接调引擎）：  
   `test/minimal_engine_test.py` 中首步显式传入完整 `input_ids` 做 prefill，之后每步只传上一 token。运行：
   ```bash
   PYTHONPATH=. .venv/bin/python test/minimal_engine_test.py --model /path/to/DeepSeek-R1-Distill-Qwen-1___5B --prompt "什么是数学" --max_steps 50
   ```
   可得到正常中文续写（如「嗯，数学是什么呢？让我好好想想。数学…」），证明引擎与权重正常。

2. **修复后**：重启服务，通过聊天界面或 `/v1/chat/completions` 流式请求同一 prompt，回复恢复正常，不再出现数字串。

### 11.5 小结

| 项目     | 说明 |
|----------|------|
| **根因** | 流式首步误传 `tokens[-1:]`，未对完整 prompt 做 prefill。 |
| **修改** | 首步（`next_id is None` 且 `len(tokens)>1`）改为传完整 `tokens`。 |
| **影响** | 仅影响流式 SSE 路径；非流式 `generate()` 本身逻辑正确，未改。 |
