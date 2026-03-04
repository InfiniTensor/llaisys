# LLAISYS 优化进度记录

## 1. 目标
- 持续优化 NVIDIA 推理路径（优先 `linear`、`self_attention`、Qwen2 decode 路径）。
- 在保证正确性的前提下，缩小与 Torch的时延差距。
- 每次改动都记录：假设 -> 改动 -> 测试 -> 结论 -> 下一步。

## 2. 记录规则（每一步都按此格式）
- `Step ID`：递增编号（S001, S002 ...）。
- `日期`：YYYY-MM-DD。
- `目标`：本步优化对象（算子/调度/缓存/内存/构建）。
- `假设`：为什么这步可能提升性能。
- `改动文件`：列出具体路径。
- `测试命令`：可复现命令。
- `结果`：关键指标（time/ms, ratio, tokens/s）。
- `结论`：是否有效，是否保留。
- `下一步`：基于结果的后续动作。
- 回退策略（强制）：若本步在统一口径下无正向收益（性能持平或变慢），则必须回退该步代码，仅保留实验记录。

---

## 3. 当前统一测试命令（基准）
### 3.1 算子级
```bash
python test/ops/linear.py --device nvidia --profile
python test/ops/self_attention.py --device nvidia --profile
```

### 3.2 端到端
```bash
python test/test_infer.py --model /home/wgreymon/model_pkg/DeepSeek-R1-Distill-Qwen-1.5B/ --device nvidia --test
python test/test_infer.py --model /home/wgreymon/model_pkg/DeepSeek-R1-Distill-Qwen-1.5B/ --device nvidia --max_steps 32
```

---

## 4. 基线记录（首次）
> 注：以下为 2026-03-02 的一次完整复测结果，后续建议同命令至少 3 次取中位数。

| 场景 | Torch | LLAISYS | 备注 |
|---|---:|---:|---|
| linear f32, (512,4096)x(4096,4096) | 2.70780ms | 2.05755ms | 本次 LLAISYS 更快 |
| linear f16, (512,4096)x(4096,4096) | 0.60095ms | 0.58783ms | 接近持平 |
| linear bf16, (512,4096)x(4096,4096) | 0.55254ms | 0.58733ms | LLAISYS 略慢 |
| self_attention f32, qlen=5 kvlen=11 nh=4 nkvh=2 hd=8 | 0.61596ms | 0.03589ms | 小规模 shape，LLAISYS 更快 |
| self_attention f16, qlen=5 kvlen=11 nh=4 nkvh=2 hd=8 | 0.61107ms | 0.03487ms | 小规模 shape，LLAISYS 更快 |
| self_attention bf16, qlen=5 kvlen=11 nh=4 nkvh=2 hd=8 | 0.60352ms | 0.05624ms | 小规模 shape，LLAISYS 更快 |
| test_infer --test | 通过 | 通过 | token 对齐 |

---

## 5. 优化日志

### S001
- 日期：2026-03-02
- 目标：建立统一优化日志与流程
- 假设：统一记录可减少重复试错，提升后续优化效率
- 改动文件：`OPTIMIZATION_PROGRESS.md`
- 测试命令：N/A
- 结果：日志模板已建立
- 结论：保留
- 下一步：进入 S002，先做一次“完整基线复测”并填入本页

### S002
- 日期：2026-03-02
- 目标：执行统一基线复测并固化结果
- 假设：先拿到同环境可复现实测数据，后续优化才能做有效对比
- 改动文件：`OPTIMIZATION_PROGRESS.md`
- 测试命令：
  - `python test/ops/linear.py --device nvidia --profile`
  - `python test/ops/self_attention.py --device nvidia --profile`
- 结果：
  - linear/f32: Torch `2.70780ms`, LLAISYS `2.05755ms`
  - linear/f16: Torch `0.60095ms`, LLAISYS `0.58783ms`
  - linear/bf16: Torch `0.55254ms`, LLAISYS `0.58733ms`
  - self_attention 测试集全部通过，测得 LLAISYS 在当前小规模 case 显著快于 Torch
- 结论：保留；当前热点优先从 `bf16 linear` 和端到端 decode 路径继续深挖
- 下一步：进入 S003，补跑端到端 `test_infer` 基线并拆分算子占比

### S003
- 日期：2026-03-02
- 目标：降低端到端 decode 时延（减少重复分配与无效 kernel）
- 假设：
  - decode 阶段大量 `Tensor::create` 触发频繁 `cudaMalloc/cudaFree`，会显著拖慢
  - 无 bias 的 linear 传入 dummy bias 会触发额外 `add_bias` kernel，属于纯开销
- 改动文件：
  - `src/models/qwen2/model.hpp`
  - `src/models/qwen2/model.cpp`
- 测试命令：
  - `xmake && xmake install`
  - `python test/test_infer.py --model /home/wgreymon/model_pkg/DeepSeek-R1-Distill-Qwen-1.5B/ --device nvidia --max_steps 32`
  - `python test/test_infer.py --model /home/wgreymon/model_pkg/DeepSeek-R1-Distill-Qwen-1.5B/ --device nvidia --test`
- 结果：
  - `--max_steps 32`: LLAISYS `9.28s -> 8.74s`
  - `--test`: LLAISYS `24.49s -> 23.20s`
  - token 对齐保持通过（`Test passed`）
- 结论：有效但收益中等；说明当前主瓶颈已转向 decode 小 batch 的 kernel 启动/算子粒度问题
- 下一步：进入 S004，增加层级 profile（linear/self_attention/rms_norm/rope/swiglu 占比）

### S004
- 日期：2026-03-02
- 目标：实现 allocator 缓存池，减少 decode 高频分配抖动
- 假设：`malloc/free` 改为缓存池后，端到端推理时延会明显下降
- 改动文件：
  - `src/core/allocator/naive_allocator.hpp`
  - `src/core/allocator/naive_allocator.cpp`
- 测试命令：
  - `xmake && xmake install`
  - `python test/test_runtime.py --device nvidia`
  - `python test/test_infer.py --model /home/wgreymon/model_pkg/DeepSeek-R1-Distill-Qwen-1.5B/ --device nvidia --max_steps 32`
  - `python test/test_infer.py --model /home/wgreymon/model_pkg/DeepSeek-R1-Distill-Qwen-1.5B/ --device nvidia --test`
- 结果：
  - runtime 测试通过
  - `--max_steps 32`: `8.74s -> 8.79s`（波动范围内，近似无提升）
  - `--test`: `23.20s -> 23.30s`（波动范围内，近似无提升）
  - token 对齐通过（`Test passed`）
- 结论：本步对端到端收益很小；说明当前瓶颈主要不在 allocator，而在 decode 小算子/attention kernel 粒度
- 下一步：S005 只做一项：`seqlen=1` 专用 attention kernel 或先加层级 profile（二选一）

### S005
- 日期：2026-03-02
- 目标：引入层级 profile，定位端到端热点占比
- 假设：先用数据确认热点，再决定下一步优化对象，避免盲改
- 改动文件：
  - `src/models/qwen2/model.hpp`
  - `src/models/qwen2/model.cpp`
- 说明：
  - 通过环境变量开关 profile：`LLAISYS_PROFILE=1`
  - 统计项覆盖：embedding、每层 linear/attn/rope/rms/swiglu/add、out_linear、argmax
  - profile 模式对每个算子后同步，绝对值会偏大，主要看占比
- 测试命令：
  - `xmake && xmake install`
  - `LLAISYS_PROFILE=1 python test/test_infer.py --model /home/wgreymon/model_pkg/DeepSeek-R1-Distill-Qwen-1.5B/ --device nvidia --max_steps 32`
- 结果：
  - 端到端：`Time elapsed: 9.51s`
  - 层内占比（layer_breakdown）：
    - `linear`: `94.525%`
    - `attn`: `0.651%`
    - `rope`: `1.022%`
    - `rms`: `1.089%`
    - `swiglu`: `0.941%`
    - `add`: `1.772%`
- 结论：当前 decode 主瓶颈非常明确在 `linear`（远高于 attention）；下一步应优先减少 linear 次数（QKV 融合、Gate/Up 融合）
- 下一步：S006 只做一项：实现 QKV 融合 linear（先不动 attention kernel）

### S006
- 日期：2026-03-02
- 目标：decode 路径 QKV 融合（每层 `3x linear -> 1x linear`）
- 假设：`seqlen=1` 下 kernel launch 开销显著，减少 linear 调用次数可降低端到端时延
- 改动文件：
  - `src/models/qwen2/model.hpp`
  - `src/models/qwen2/model.cpp`
- 改动说明：
  - 新增每层 QKV fused weight/bias 缓存（按 `[Q;K;V]` 拼接）。
  - 仅在 `seqlen==1` 时走 fused 路径；prefill 仍走原始三次 linear。
  - fused 输出拆分回连续 `q_flat_/k_flat_/v_flat_` 供后续 rope/attention 复用。
- 测试命令：
  - `xmake && xmake install`
  - `python test/test_infer.py --model /home/wgreymon/model_pkg/DeepSeek-R1-Distill-Qwen-1.5B/ --device nvidia --max_steps 32`
  - `python test/test_infer.py --model /home/wgreymon/model_pkg/DeepSeek-R1-Distill-Qwen-1.5B/ --device nvidia --test`
- 结果：
  - `--max_steps 32`: `8.89s`, `8.91s`（与 S005 的 `8.79s` 基本持平/略慢）
  - `--test`: `23.72s`（对比 S005 的 `23.30s`，无明显提升）
  - 正确性：`Test passed`
- 结论：
  - 该步在当前实现下收益不明显，可能被“fused 输出拆分拷贝 + 首次 fused 权重拼接开销”抵消。
  - 当前仍应优先针对 `linear` 做 decode 专用高效路径，而不是仅在模型层做调用合并。
- 下一步：S007 只做一项：为 `ops::linear` 增加 decode 形状（`M=1`）专用 fast path（优先调用 cuBLAS/cuBLASLt）

### S006-补充分析（算子级复测）
- 日期：2026-03-02
- 目标：确认端到端慢是否来自 `linear` 算子本身性能不足
- 假设：若单算子与 Torch 接近，则端到端瓶颈更可能来自 decode 阶段“调用次数/调度开销”
- 改动文件：
  - `OPTIMIZATION_PROGRESS.md`
- 测试命令：
  - `python test/ops/linear.py --device nvidia --profile`
- 结果（用户复测）：
  - 小形状：
    - f32 `(2,3)x(3,4)`: Torch `0.01766ms`, LLAISYS `0.01127ms`
    - f16 `(2,3)x(3,4)`: Torch `0.01236ms`, LLAISYS `0.01153ms`
    - bf16 `(2,3)x(3,4)`: Torch `0.01167ms`, LLAISYS `0.01200ms`
  - 大形状：
    - f32 `(512,4096)x(4096,4096)`: Torch `1.95276ms`, LLAISYS `2.01260ms`
    - f16 `(512,4096)x(4096,4096)`: Torch `0.57978ms`, LLAISYS `0.58821ms`
    - bf16 `(512,4096)x(4096,4096)`: Torch `0.55290ms`, LLAISYS `0.58798ms`
- 结论：
  - 单次 `linear` 性能与 Torch 已较接近，差距不足以解释端到端 `test_infer` 的大幅时延差。
  - 结合 S005（`linear` 占层内约 `94.5%`）可判定：当前核心问题是 decode 阶段 `linear` 调用数量过多 + 小算子 launch/调度开销累计。
  - 优化重点应放在“减少调用次数/融合算子/decode 执行图复用”，而不是简单替换 `linear` 后端。
- 下一步：S007 只做一项：实现 `gate + up + swiglu` 融合路径（先在 decode `seqlen=1` 启用）

### S007
- 日期：2026-03-02
- 目标：实现 decode 路径 `gate+up` 融合 linear（每层 `2x linear -> 1x linear`）
- 假设：减少一半 MLP 前半段 linear 调用，可降低 decode 小算子 launch 开销
- 改动文件：
  - `src/models/qwen2/model.hpp`
  - `src/models/qwen2/model.cpp`
- 改动说明：
  - 新增每层融合权重 `mlp_gate_up_w_`（`[gate;up]` 拼接）。
  - 仅在 `seqlen==1` 启用 fused 路径；prefill 保持原实现。
  - fused 输出复制拆分到连续 `gate_` / `up_`，复用现有 `swiglu` 接口。
- 测试命令：
  - `xmake && xmake install`
  - `python test/test_infer.py --model /home/wgreymon/model_pkg/DeepSeek-R1-Distill-Qwen-1.5B/ --device nvidia --max_steps 32`
  - `python test/test_infer.py --model /home/wgreymon/model_pkg/DeepSeek-R1-Distill-Qwen-1.5B/ --device nvidia --test`
- 结果：
  - `--max_steps 32`: `9.62s`, `9.70s`（较 S006 `8.89s/8.91s` 明显变慢）
  - `--test`: `25.67s`（较 S006 `23.72s` 变慢）
  - 正确性：`Test passed`
- 结论：
  - 当前实现收益为负，主要被“fused 输出拆分复制 + 更大形状单次 GEMM 调度特性”抵消。
  - 在不改 `swiglu` 接口/内核的前提下，此融合路径不建议保留。
- 状态：已回退（恢复到 S006 的 MLP 路径）
- 下一步：S008 只做一项：`M=1` decode CUDA Graph（捕获整步 decode）单步验证

### S008
- 日期：2026-03-02
- 目标：降低 decode 主机端开销（减少高频 `slice` 临时对象）
- 假设：每层每步频繁创建 `Tensor::slice`（KV cache update + attention 输入）会产生可见 CPU 开销；改为“整块 cache + total_len 参数”可降低开销
- 改动文件：
  - `src/models/qwen2/model.cpp`
  - `src/ops/self_attention/op.hpp`
  - `src/ops/self_attention/op.cpp`
- 改动说明：
  - `update_kv_cache` 改为直接按字节偏移写入 cache（不再构造 `k_slice/v_slice`）。
  - `ops::self_attention` 增加 `total_len_override` 参数，允许传入整块 KV cache + 真实 `total_len`。
  - `forward_layer` 不再对 cache 做 `slice(0, 0, total_len)`，直接调用 attention 覆盖长度参数。
- 测试命令：
  - `xmake && xmake install`
  - `python test/test_infer.py --model /home/wgreymon/model_pkg/DeepSeek-R1-Distill-Qwen-1.5B/ --device nvidia --max_steps 32`
  - `python test/test_infer.py --model /home/wgreymon/model_pkg/DeepSeek-R1-Distill-Qwen-1.5B/ --device nvidia --test`
- 结果：
  - `--max_steps 32`: `8.85s`（对比回退后 S006 态 `8.80s`，近似持平/略慢）
  - `--test`: `22.65s`（对比回退后 S006 态 `22.47s`，近似持平/略慢）
  - 正确性：`Test passed`
- 结论：
  - 该步对端到端收益不明显，说明 decode 主瓶颈仍主要在 GPU 小算子 launch/调度侧，而非这些主机对象创建。
  - 已按“无收益即回退”原则回退该步代码，保持主分支简洁。
- 状态：已回退（恢复到 S006 稳定状态）
- 下一步：S009 只做一项：实现 `decode(seqlen=1)` 的阶段化时间分解（Host prepare / GPU forward / D2H argmax），先量化“主机 vs 设备”占比

### S009
- 日期：2026-03-02
- 目标：验证 `M==1` decode 线性层 fast path（f32 用 `cublasSgemv`）
- 假设：decode 常见 `M=1`，`sgemm -> sgemv` 可降低该场景的调度开销
- 改动文件（实验分支）：
  - `src/ops/linear/nvidia/linear_nvidia.cu`
  - `test/ops/linear.py`（临时加入 `M=1` 基准 case）
- 测试命令：
  - `xmake && xmake install`
  - `python test/ops/linear.py --device nvidia --profile`
  - `python test/test_infer.py --model /home/wgreymon/model_pkg/DeepSeek-R1-Distill-Qwen-1.5B/ --device nvidia --max_steps 32`
  - `python test/test_infer.py --model /home/wgreymon/model_pkg/DeepSeek-R1-Distill-Qwen-1.5B/ --device nvidia --test`
- 结果：
  - 算子级（临时 `M=1` case）：
    - f32: Torch `0.25836ms`, LLAISYS `0.26067ms`
    - f16: Torch `0.04603ms`, LLAISYS `0.04785ms`
    - bf16: Torch `0.04536ms`, LLAISYS `0.04729ms`
  - 端到端：
    - `--max_steps 32`: `10.10s`, `9.03s`（对比基线 `~8.85s`，无收益）
    - `--test`: `23.89s`（对比基线 `~22.7s`，无收益）
  - 正确性：`Test passed`
- 结论：
  - 该方案在当前实现下无正向收益，端到端有退化。
  - 按“无收益即回退”规则已回退全部 S009 代码改动。
- 状态：已回退（代码恢复到 S008 回退后的稳定版本）
- 下一步：S010 只做一项：增加 decode 分阶段计时（Host prepare / forward / argmax D2H），先定量定位剩余瓶颈

### S010
- 日期：2026-03-02
- 目标：实现 decode 分阶段计时，定量拆分 `Host prepare / forward / argmax / D2H`
- 假设：先量化 decode 各阶段占比，避免继续在低占比环节投入
- 改动文件：
  - `src/models/qwen2/model.hpp`
  - `src/models/qwen2/model.cpp`
- 改动说明：
  - 在 `infer(ntoken==1)` 下新增分阶段统计字段与计时：
    - `profile_decode_host_prepare_ms_`
    - `profile_decode_forward_ms_`
    - `profile_decode_argmax_ms_`
    - `profile_decode_d2h_ms_`
  - 在 profile 汇总中新增 `decode_stage(ms)` 与 `decode_stage_avg_per_step(ms)` 输出。
  - 非 profile 路径逻辑不变。
- 测试命令：
  - `xmake && xmake install`
  - `LLAISYS_PROFILE=1 python test/test_infer.py --model /home/wgreymon/model_pkg/DeepSeek-R1-Distill-Qwen-1.5B/ --device nvidia --max_steps 32`
  - `python test/test_infer.py --model /home/wgreymon/model_pkg/DeepSeek-R1-Distill-Qwen-1.5B/ --device nvidia --max_steps 32`
  - `python test/test_infer.py --model /home/wgreymon/model_pkg/DeepSeek-R1-Distill-Qwen-1.5B/ --device nvidia --test`
- 结果：
  - profile 分阶段（decode 31 步）：
    - `host_prepare=0.339ms (0.004%)`
    - `forward=9150.830ms (99.766%)`
    - `argmax=16.333ms (0.178%)`
    - `d2h=4.746ms (0.052%)`
    - 每步均值：`host=0.011ms, forward=295.188ms, argmax=0.527ms, d2h=0.153ms`
  - 非 profile 回归：
    - `--max_steps 32`: `9.88s`（一次抖动）、`8.88s`（与基线 `~8.9s` 一致）
    - `--test`: `22.69s`, `Test passed`
- 结论：
  - decode 主耗时几乎全部在 `forward`（GPU 计算段），主机准备与 D2H 占比可忽略。
  - 后续优化应集中在 `forward` 内部，尤其 `linear` 与 `out_linear(lm_head)` 的 decode 路径。
- 状态：保留（观测能力增强，非 profile 路径无行为变化）
- 下一步：S011 只做一项：针对 `lm_head(out_linear)` 的 `M=1` 专用路径做单点优化并 A/B（无收益即回退）

### S011
- 日期：2026-03-02
- 目标：重试 `gate+up` 融合，但去掉中间拆分拷贝（直接用 fused buffer 两段指针计算 SwiGLU）
- 假设：若不做 D2D 拆分复制，则 `2x linear -> 1x linear` 可能带来 decode 提升
- 改动文件（实验分支）：
  - `src/models/qwen2/model.hpp`
  - `src/models/qwen2/model.cpp`
- 改动说明：
  - 新增 `mlp_gate_up_w_` 融合权重缓存。
  - decode `seqlen=1 && nvidia` 路径：先做 fused linear 得到 `[1, 2*di]`，再直接以两段指针调用 `nvidia::swiglu`（无中间复制）。
- 测试命令：
  - `xmake && xmake install`
  - `python test/test_infer.py --model /home/wgreymon/model_pkg/DeepSeek-R1-Distill-Qwen-1.5B/ --device nvidia --max_steps 32`
  - `python test/test_infer.py --model /home/wgreymon/model_pkg/DeepSeek-R1-Distill-Qwen-1.5B/ --device nvidia --test`
- 结果：
  - `--max_steps 32`: `9.63s`, `10.76s`（基线约 `8.9s`，显著变慢）
  - `--test`: `25.44s`（基线约 `22.7s`，变慢）
  - 正确性：`Test passed`
- 结论：
  - 该方案仍无收益，且退化明显。
  - 按“无收益即回退”规则，已回退全部 S011 代码改动。
- 状态：已回退（恢复到 S010 稳定版本）
- 下一步：S012 只做一项：针对 `lm_head(out_linear)` 做实验（例如 cublasLt/分块 top1 路径）并严格 A/B

### S015
- 日期：2026-03-02
- 目标：减少端到端 `malloc/free` 开销（Lazy Allocation / 张量复用）
- 假设：避免每步 `Tensor::create` 带来的开销，复用成员变量张量
- 改动文件：
  - `src/models/qwen2/model.hpp`
  - `src/models/qwen2/model.cpp`
- 改动说明：
  - 将 `x_`, `q_`, `k_` 等中间张量提升为成员变量。
  - 在 `forward` 和 `forward_layer` 中添加 `if (!ptr || shape_mismatch) ptr = create(...)` 逻辑。
- 测试命令：
  - `xmake && xmake install`
  - `python test/test_infer.py --model /home/wgreymon/model_pkg/DeepSeek-R1-Distill-Qwen-1.5B/ --device nvidia --max_steps 32`
- 结果：
  - `--max_steps 32`: `8.9s -> 10.3s` (变慢)
  - 可能是因为条件判断开销，或者破坏了某些缓存局部性，且原先的分配并非主要瓶颈。
- 结论：
  - 无正向收益，已回退。
  - **重要调整**：后续测试增加与 Torch 的比值 (Ratio) 观测，避免仅看绝对时间。
  - 当前基线 (S015 回退后): LLAISYS/Torch Ratio ≈ 7.57x (LLAISYS 12.01s / Torch 1.59s) - *注：Torch 极快可能是因为 warmup 或 cache，需关注相对变化*
- 状态：已回退
- 下一步：S016 针对 `M=1` 的 Linear 算子进行优化（重点关注 `N` 较大的情况，如 `lm_head`）。

---

## 6. 待办队列（优先级）
- [ ] P0：复测并固化基线（同一机器、同一命令、至少 3 次取中位数）
- [ ] P1：定位 `f16/bf16 linear` 与 Torch 差距（kernel 实现路径 vs cublasLt 路径）
- [ ] P2：`self_attention` decode 专用小 batch / seqlen=1 路径
- [ ] P3：Qwen2 端到端热点拆分（linear/attn/rope/rms_norm 占比）
- [ ] P4：引入统一 profile 输出（每层耗时 + 累计占比）

---

## 7. 当前主要问题与解决方案（2026-03-02）
- 问题A：decode 阶段 `seqlen=1`，小算子数量多，kernel launch/调度开销占比高。
  - 方案A1：增加 decode 专用路径（`ntoken==1`），减少通用路径中的冗余逻辑。
  - 方案A2：融合线性层（QKV 合并、Gate/Up 合并）减少 kernel 次数。
  - 方案A3：条件允许时引入 CUDA Graph 复用 decode 执行图。

- 问题B：attention kernel 仍偏通用实现，decode 形状下并不高效。
  - 方案B1：新增 `qlen=1` 专用 attention kernel（多 warp 并行扫 K tile）。
  - 方案B2：`f16/bf16` 路径使用 `half2/bfloat162` 向量化访存与计算。
  - 方案B3：保留通用 kernel 作为回退，按 shape 自动分发。

- 问题C：当前 allocator 为直连 `malloc/free`，没有缓存池。
  - 方案C1：实现 size-class 缓存池分配器，`release` 回收至池而非立即 free。
  - 方案C2：runtime 析构时统一释放池中内存，避免长期泄漏。
  - 方案C3：对 decode 高频 shape 做内存复用，减少分配抖动。

优化执行顺序（高收益优先）：
1. C（allocator 缓存池）
2. A（线性层融合 + decode 专用路径）
3. B（decode 专用 attention kernel）

---

## 8. 单步记录模板（复制追加）

### SXXX
- 日期：
- 目标：
- 假设：
- 改动文件：
- 测试命令：
- 结果：
- 结论：
- 下一步：

---

## 9. 重新梳理的优化顺序（2026-03-02）

已确认的主结论：
- decode 阶段耗时几乎都在 `forward`。
- `forward` 内部主要瓶颈是 `linear`，其中 `out_linear(lm_head)` 是大头之一。
- 继续在低占比环节（host_prepare/D2H/attention）优化，端到端收益有限。

后续优化顺序（严格单步 A/B，无收益即回退）：
1. `lm_head(out_linear)` 专项（`M=1, N=vocab` 形状）
2. 减少 decode 的 `linear` 调用次数（优先 grouped/融合）
3. decode CUDA Graph（降低小算子 launch 开销）
4. 低优先级：attention decode 专用 kernel 与 allocator 深挖

### S012（已回退）
- 日期：2026-03-02
- 目标：降低 decode 中 KV cache 写回的主机阻塞开销
- 假设：`update_kv_cache` 每层每步 2 次 `memcpy_sync(D2D)` 会造成高频 host wait，改为同流 `memcpy_async` 可减少阻塞
- 改动文件：
  - `src/models/qwen2/model.cpp`
- 改动说明：
  - `update_kv_cache` 中两处 D2D 拷贝由 `memcpy_sync` 改为 `memcpy_async(..., nullptr)`（默认 stream）
- 测试命令：
  - `xmake && xmake install`
  - `python test/test_infer.py --model /home/wgreymon/model_pkg/DeepSeek-R1-Distill-Qwen-1.5B/ --device nvidia --max_steps 32`
  - `python test/test_infer.py --model /home/wgreymon/model_pkg/DeepSeek-R1-Distill-Qwen-1.5B/ --test --device nvidia`
- 结果：
  - 出现运行时崩溃（`Segmentation fault`），无法稳定完成端到端测试。
- 结论：
  - 该方案不稳定，不满足“先正确再提速”的要求。
  - 已按“无收益或不稳定即回退”原则回退 `model.cpp` 对应改动。
- 下一步：S013 先做“可控定位”而非继续盲目改算子。

### S013（进行中）
- 日期：2026-03-02
- 目标：定位当前不稳定/性能波动是否由 allocator 内存池引入
- 假设：若关闭池化后稳定性显著提升，则优先修 allocator；否则继续 `lm_head` 专项
- 改动文件：
  - `src/core/allocator/naive_allocator.hpp`
  - `src/core/allocator/naive_allocator.cpp`
- 改动说明：
  - allocator 策略改为“默认直连 `malloc/free`（禁用池化）”
  - 新增环境变量：`LLAISYS_ALLOCATOR_ENABLE_POOL=1` 时才启用池化
  - 目的：先保证稳定性，再做性能 A/B
- 测试命令：
  - `xmake && xmake install`
  - `python test/test_infer.py --model /home/wgreymon/model_pkg/DeepSeek-R1-Distill-Qwen-1.5B/ --device nvidia --max_steps 32`
  - `LLAISYS_ALLOCATOR_ENABLE_POOL=1 python test/test_infer.py --model /home/wgreymon/model_pkg/DeepSeek-R1-Distill-Qwen-1.5B/ --device nvidia --max_steps 32`
  - `python test/test_infer.py --model /home/wgreymon/model_pkg/DeepSeek-R1-Distill-Qwen-1.5B/ --test --device nvidia`
  - `LLAISYS_ALLOCATOR_ENABLE_POOL=1 python test/test_infer.py --model /home/wgreymon/model_pkg/DeepSeek-R1-Distill-Qwen-1.5B/ --test --device nvidia`
- 结果：待补充
- 结果：
  - `LLAISYS_ALLOCATOR_ENABLE_POOL=0/1` 两种模式均出现 `Segmentation fault (exit=139)`。
- 结论：
  - 崩溃与 allocator 池化开关无关，需转向其他改动点排查。
- 下一步：S014 做风险隔离：默认关闭 decode QKV 融合路径，仅在环境变量显式开启时使用。

### S014（进行中）
- 日期：2026-03-02
- 目标：快速恢复稳定性，隔离 decode QKV 融合路径是否为崩溃源
- 假设：`seqlen=1` 下的 QKV 融合路径可能引入了非法访存/生命周期问题
- 改动文件：
  - `src/models/qwen2/model.hpp`
  - `src/models/qwen2/model.cpp`
- 改动说明：
  - 新增环境变量 `LLAISYS_ENABLE_DECODE_QKV_FUSED`
  - 默认关闭 decode QKV 融合；仅当显式设为 1 时启用
- 测试命令：
  - `xmake && xmake install`
  - `timeout 180s env PYTHONUNBUFFERED=1 LLAISYS_PROFILE=1 python test/test_infer.py --model /home/wgreymon/model_pkg/DeepSeek-R1-Distill-Qwen-1.5B/ --device nvidia --max_steps 32 > /tmp/s014_default.log 2>&1; echo DEFAULT_EXIT:$?`
  - `timeout 180s env PYTHONUNBUFFERED=1 LLAISYS_PROFILE=1 LLAISYS_ENABLE_DECODE_QKV_FUSED=1 python test/test_infer.py --model /home/wgreymon/model_pkg/DeepSeek-R1-Distill-Qwen-1.5B/ --device nvidia --max_steps 32 > /tmp/s014_fused.log 2>&1; echo FUSED_EXIT:$?`
  - `rg -n "Time elapsed|Test passed|Segmentation fault" /tmp/s014_default.log /tmp/s014_fused.log`
- 结果：待补充
- 结论：待补充
- 下一步：若默认模式恢复稳定，则在稳定基线继续 `lm_head` 专项优化

---

## 10. Source Control 审计（2026-03-02）

当前 `git status --short`：
- `D matmul_optimization_summary_kimi.md`
- `M src/core/allocator/naive_allocator.hpp`
- `M src/core/allocator/naive_allocator.cpp`
- `?? OPTIMIZATION_PROGRESS.md`

关键结论：
- 目前代码改动仅集中在 allocator；`model.cpp/model.hpp` 没有未提交改动。
- 与上文 `S014` 的“已改 model 代码”描述不一致，说明该步骤尚未落地到当前工作树。
- 现阶段端到端 `exit=139` 不能直接归因于 allocator 池化开关（开/关均崩）。

后续执行原则（重置）：
1. 先恢复“稳定可运行基线”再做性能优化。
2. 每一步只改一个点，跑固定命令，记录 `exit code + time`。
3. 无收益或不稳定立即回退。

---

## 11. 计划重置（2026-03-02）

背景：
- 已回退到较早稳定代码形态，`model.cpp` 不含此前的 profile/复用/QKV 融合逻辑。
- 当前优化目标是“先稳定、再提速”，避免再次进入不可定位的崩溃状态。

### 总体策略
1. **可观测优先**：先恢复最小 profile，确保每步优化有数据支撑。
2. **低风险优先**：先做不改算子数学逻辑的改动（消除无效 kernel、减少临时分配）。
3. **单点实验**：每次只改一个点，固定命令 A/B，失败立即回退。
4. **阶段门禁**：不稳定（crash/错误）优先修复，停止后续性能优化。

### 分阶段计划

#### P0：稳定性与基线（必须先过）
- 目标：保证 `test_infer` 可稳定运行并具备可比较基线。
- 动作：
  - 固定测试命令与日志路径。
  - 记录 3 次 `--max_steps 32` 中位数。
- 验收：
  - 无 `Segmentation fault`。
  - `--test` 正确性通过。

#### P1：低风险减开销（结构不变）
- 目标：减少不必要 kernel launch 与内存流量。
- 子步骤：
  - S100：去掉无效 bias 路径（zero-bias 线性传 `nullptr`）。
  - S101：恢复 `ensure_tensor` 缓冲复用（先 infer 入口与输出张量）。
  - S102：扩展复用到 layer 内高频临时张量。
- 验收：
  - 正确性不变；
  - `--max_steps 32` 中位数有正收益。

#### P2：decode 专用路径
- 目标：针对 `seqlen=1` 降低固定开销。
- 子步骤：
  - S200：decode 路径减少 `slice/view/create` 对象构造。
  - S201：KV cache 写回改为偏移拷贝（保持同步拷贝，先稳）。
- 验收：同上。

#### P3：减少 linear 调用次数
- 目标：在不破坏稳定性的前提下降低 decode launch 数量。
- 子步骤：
  - S300：QKV grouped/fused（默认关闭，环境变量开关）。
  - S301：gate/up grouped（同上）。
- 验收：仅在 A/B 显著收益时保留。

#### P4：CUDA Graph（decode-only）
- 目标：进一步降低 launch overhead。
- 前置条件：
  - P1/P2 后无崩溃，shape/控制流足够稳定。
- 验收：
  - `--max_steps 32` 有稳定收益；
  - 不破坏 `--test`。

### 固定测试命令模板（每步统一）
- 构建：
  - `xmake && xmake install`
- 性能（至少 3 次取中位）：
  - `python test/test_infer.py --model /home/wgreymon/model_pkg/DeepSeek-R1-Distill-Qwen-1.5B/ --device nvidia --max_steps 32`
- 正确性：
  - `python test/test_infer.py --model /home/wgreymon/model_pkg/DeepSeek-R1-Distill-Qwen-1.5B/ --test --device nvidia`

### 回退规则（强制）
- 出现 crash / correctness fail：立即回退该步。
- 性能无提升或抖动不可区分：回退该步。

### S100（已回退）
- 目标：去掉无效 bias kernel，降低 decode 的固定 launch 开销。
- 假设：o_proj / mlp_gate / mlp_up / mlp_down / out_embed 的 bias 实际为零，传 `dummy_bias_*` 会触发多余 add-bias kernel。
- 改动文件：
  - `src/models/qwen2/model.cpp`
- 验证命令：使用“固定测试命令模板”。
- 已完成代码改动：
  - `attn_o_w` linear：`dummy_bias_hs_ -> nullptr`
  - `mlp_gate_w` linear：`dummy_bias_di_ -> nullptr`
  - `mlp_up_w` linear：`dummy_bias_di_ -> nullptr`
  - `mlp_down_w` linear：`dummy_bias_hs_ -> nullptr`
  - `out_embed` linear：`dummy_bias_voc_ -> nullptr`
- 结果：
  - `--test` 连续两次：
    - `Time elapsed: 25.26s`
    - `Time elapsed: 26.26s`
  - 正确性：`Test passed`
- 结论：
  - 相比此前稳定区间（约 23~25s）未观察到正向收益，且有轻微退化趋势。
  - 按“无收益即回退”规则，已回退 `src/models/qwen2/model.cpp` 的本步改动。
- 下一步：
  - 进入 `S101`：仅做张量复用（`ensure_tensor`）的最小改动，先从 `infer` 入口与 `forward` 输出张量开始，避免一次性大改。

### S101（已回退）
- 目标：通过张量复用减少 decode 阶段频繁 `Tensor::create` 带来的分配/析构开销。
- 假设：`infer` 每步会重复创建输入/argmax 张量；`forward` 每步会重复创建 `x_ / x_norm_ / logits_`，可通过 `ensure_tensor` 复用。
- 改动文件：
  - `src/models/qwen2/model.hpp`
  - `src/models/qwen2/model.cpp`
- 已完成代码改动（已回退）：
  - 新增 `ensure_tensor(...)` 辅助函数。
  - 复用 `x_ / x_norm_ / logits_`。
  - 复用 `input_ids` 输入缓存、`argmax` 输出缓存。
- 验证命令：使用“固定测试命令模板”中的正确性命令。
- 结果（`--test` 连续两次）：
  - `Time elapsed: 24.74s`
  - `Time elapsed: 25.52s`
  - 正确性：`Test passed`
- 结论：
  - 相比基线（`25.26s / 26.26s`）没有形成稳定、可复现的收益（波动区间内）。
  - 按“无收益即回退”规则，已回退 `src/models/qwen2/model.cpp` 与 `src/models/qwen2/model.hpp` 的本步改动。
- 下一步：
  - 进入 `S102` 前先补充更细粒度 profile（按算子/阶段拆分），确认真实瓶颈再做下一轮最小改动。

### S102（已保留）
- 目标：扩展张量复用到 `forward_layer` 的高频临时张量，减少 decode 每层 `Tensor::create` 次数。
- 假设：单步 decode 的瓶颈之一是大量小张量重复分配/释放（每层多次），将其改为 `ensure_tensor` 复用可降低 runtime 开销。
- 改动文件：
  - `src/models/qwen2/model.hpp`
  - `src/models/qwen2/model.cpp`
- 主要改动：
  - 新增/启用 `ensure_tensor(...)`。
  - 复用层内关键临时张量：`q_/k_/v_/q_rope_/k_rope_new_/attn_out_/attn_proj_out_/x_attn_/gate_/up_/swiglu_out_/mlp_out_/x_mlp_`。
  - 复用 forward/infer 张量：`x_/x_norm_/logits_/pos_ids_q_/input_ids_buf_/max_idx_/max_val_`。
  - `Q/K/V` 改为“3D 缓冲 + 2D view 输出”，避免每层新建 `q_flat/k_flat/v_flat` 存储。
- 验证命令：
  - `xmake -r && xmake install`
  - `python test/test_infer.py --model /home/wgreymon/model_pkg/DeepSeek-R1-Distill-Qwen-1.5B/ --test --device nvidia`
  - 再跑一次同命令确认稳定性。
- 结果：
  - 改前基线（本轮测得）：`24.81s`
  - 改后第 1 次：`23.27s`
  - 改后第 2 次：`23.29s`
  - 正确性：`Test passed`
- 结论：
  - 观察到稳定正收益（约 `1.5s`，约 `6%`），本步改动保留。
- 下一步：
  - 进入 `S200`：decode 专用优化（优先 `self_attention` 的 `seqlen=1` 快路径，减少同步与无效线程）。

### S200（已回退）
- 目标：为 `self_attention` 增加 `seqlen=1` decode 快路径，减少 block 内同步和空转线程。
- 假设：decode 主要是 `seqlen=1`，使用单 warp 专用 kernel 可降低每步 attention 开销。
- 改动文件：
  - `src/ops/self_attention/nvidia/self_attention_nvidia.cu`
- 已完成代码改动（已回退）：
  - 新增 `self_attention_decode_seqlen1_kernel`（单 warp，online softmax）。
  - 在 `seqlen == 1` 且 shape 满足条件时切换到该快路径，其余走原通用 kernel。
- 验证命令：
  - `xmake -r && xmake install`
  - `python test/test_infer.py --model /home/wgreymon/model_pkg/DeepSeek-R1-Distill-Qwen-1.5B/ --test --device nvidia`（连续两次）
- 结果：
  - 改后第 1 次：`24.15s`
  - 改后第 2 次：`25.33s`
  - 参考基线（S102）：`23.27s / 23.29s`
  - 正确性：`Test passed`
- 结论：
  - 性能退化，且波动变大；按“无收益即回退”规则，已回退本步全部代码改动。
- 下一步：
  - 进入 `S201`：优先做模型侧 decode 路径的“对象构造减法”（减少 `slice/view` 与 host 临时容器创建），保持 kernel 不变。

### S201（已回退）
- 目标：在不改 kernel 的前提下减少 decode 路径对象构造与临时分配。
- 假设：`update_kv_cache` 中每层 `slice` 创建与 host 端小对象创建有可见开销。
- 改动文件：
  - `src/models/qwen2/model.cpp`
- 已尝试改动（已回退）：
  - `update_kv_cache` 从 `slice + memcpy` 改为“直接偏移 memcpy”。
  - `pos_ids_q` 在 `seqlen=1` 走标量 load，避免每步创建 `std::vector<int64_t>`。
  - `argmax` D2H 输出改为标量接收，避免每步创建长度 1 的 vector。
- 结果：
  - 第 1 次：`Time elapsed: 27.28s`（明显慢于 S102 区间）。
  - 第 2 次：出现异常长时间运行（>2min，手动终止）。
  - 正确性：首轮 `Test passed`，但性能与稳定性不满足要求。
- 结论：
  - 判定为“无收益且不稳定”，已按规则回退本步改动，仅保留 S102。
- 下一步：
  - 进入 `S202`：先统一测量口径（固定 `--max_steps 32`，连续 3 次取中位数）再推进下一项优化，避免环境抖动导致误判。

### S103（已保留）
- 日期：2026-03-02
- 目标：恢复并保留已验证有效的张量复用优化（decode 高频临时张量 `ensure_tensor` 复用）。
- 改动文件：
  - `src/models/qwen2/model.hpp`
  - `src/models/qwen2/model.cpp`
- 结果（近期样本）：
  - `--max_steps 32`: `9.98s`, `9.86s`
  - `--test`: `Test passed`
- 结论：
  - 当前作为稳定优化基线保留，后续新实验都在此基础上进行。

### S104（已回退）
- 日期：2026-03-02
- 目标：降低 KV cache 写回的主机阻塞（`memcpy_sync -> memcpy_async`）。
- 改动文件：
  - `src/models/qwen2/model.cpp`
- 结果：
  - `--max_steps 32`: `10.04s`（相对当前区间无改善）
- 结论：
  - 未观察到稳定收益，按规则回退。

### S105（已回退）
- 日期：2026-03-02
- 目标：减少 decode 末端 argmax 调度开销（直接底层 nvidia argmax 路径 + argmax kernel 单 block 修正实验）。
- 改动文件（实验）：
  - `src/models/qwen2/model.cpp`
  - `src/ops/argmax/nvidia/argmax_nvidia.cu`
- 结果：
  - `--max_steps 32`: `9.87s`, `11.89s`（波动大、无稳定收益）
  - `--test`: `24.69s`, `Test passed`
- 结论：
  - 不满足“稳定提升”标准，已回退本步实验改动。
- 当前状态：
  - 维持 S103 基线，最近回归：`--max_steps 32 = 10.05s`, `EXIT=0`。

---

## 12. 补记（2026-03-02，遗漏项补录）

### S106（已保留）
- 日期：2026-03-02
- 目标：修复端到端对比口径，定位“同命令一次 20s+、一次 1s+”的异常波动来源。
- 假设：`test/test_infer.py` 中先跑 Torch 再跑 LLAISYS，若不释放 Torch CUDA 缓存，会对后续 LLAISYS 形成干扰，导致测得时间虚高。
- 改动文件：
  - `test/test_infer.py`
- 改动说明：
  - Torch 推理后增加：
    - `del model`
    - `gc.collect()`
    - `torch.cuda.empty_cache()`
    - `torch.cuda.synchronize()`
- 测试命令：
  - `python test/test_infer.py --model /home/wgreymon/model_pkg/DeepSeek-R1-Distill-Qwen-1.5B/ --device nvidia`
- 结果（同场景对比）：
  - 修复前样本：Torch `2.64s`，LLAISYS `25.96s`
  - 修复后样本：Torch `2.32s`，LLAISYS `1.64s`
  - token 一致性保持通过
- 结论：
  - 此前“20s+”主要是测试口径问题，不是单次代码优化带来的真实性能跳变。
  - 该修复必须长期保留，作为后续 A/B 的前置条件。
- 下一步：
  - 增加更全面、隔离后端干扰的 benchmark 脚本，作为统一对比入口。

### S107（已保留）
- 日期：2026-03-02
- 目标：建立更全面且可复现的端到端 benchmark（多 prompt、多 token 档位、多后端）。
- 假设：将 Torch/LLAISYS 分别放入独立子进程，可避免同进程资源干扰，结果更可信。
- 改动文件：
  - `test/benchmark_infer.py`
- 改动说明：
  - 新增综合 benchmark 脚本，支持：
    - `--backends`（如 `torch,llaisys`）
    - `--prompts`（`short,medium,long`）
    - `--max-new-tokens`（如 `16,32,64`）
    - `--warmup` / `--repeat`
    - `mean/p50/p95/tok-s` 指标
    - 确定性场景下的 `output_hash` 一致性对比
  - 通过 `--worker` 子进程模式运行各后端，并用 `JSON_SENTINEL` 回传结构化结果。
- 测试命令：
  - `python test/benchmark_infer.py --model /home/wgreymon/model_pkg/DeepSeek-R1-Distill-Qwen-1.5B/ --device nvidia --backends torch,llaisys --prompts short,medium,long --max-new-tokens 16,32,64 --warmup 1 --repeat 3`
- 结果：
  - 脚本可稳定输出 9 组 case 的完整报表，并支持导出 JSON。
- 结论：
  - 该脚本可作为项目二阶段性验收与后续优化对比基准，保留。
- 下一步：
  - 基于综合报表做横向分析，提炼当前性能结论和风险点。

### S108（已保留）
- 日期：2026-03-02
- 目标：解析综合 benchmark 结果，给出当前项目性能状态结论。
- 假设：覆盖不同 prompt 长度与输出 token 数后，才能判断“整体是否已接近/超过 Torch”。
- 改动文件：
  - `OPTIMIZATION_PROGRESS.md`
- 测试命令：
  - 同 S107
- 结果（9 组 case 汇总）：
  - LLAISYS 在 `8/9` 个 case 更快，仅 `long/16` 略慢（`354.30ms` vs `340.39ms`）。
  - 平均时延改善：约 `7.41%`（按 `(torch-llaisys)/torch` 的 9 case 算术平均）。
  - 平均吞吐提升：约 `8.08%`（Torch `45.23 tok/s` -> LLAISYS `48.88 tok/s`）。
  - 最优提升 case：`long/64`，时延 `1809.57ms -> 1331.86ms`（约 `26.4%` 改善）。
  - 一致性：确定性参数下有 `1` 个 case 出现 `output_match = N`（`medium/32`）。
- 结论：
  - 在当前测试口径下，LLAISYS 端到端性能已达到“总体不弱于 Torch，且多数场景更优”的状态，可视为项目二性能目标阶段性达成。
  - 仍需跟进 `medium/32` 的单例不一致问题，确认是否由边界条件或实现细节引起。
- 下一步：
  - 固化这组 benchmark 作为阶段基线（建议保存 `--json-out` 结果）。
  - 追加一个“确定性回归脚本”，专门检查 `top_k=1, top_p=1, temperature=1` 下的 token 完整一致性。

### 当前状态快照（补记）
- 统一口径后，`test/test_infer.py --device nvidia` 已不再出现“LLAISYS 首次 20s+”的误判。
- 端到端综合 benchmark 显示：LLAISYS 在多数场景已具备可用竞争力。
- 后续优化重点从“粗粒度提速”转向“确定性一致性 + 长稳态回归”。
