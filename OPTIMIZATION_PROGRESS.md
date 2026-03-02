# LLAISYS 推理框架性能优化历程记录（去重整理版）

最后更新：2026-03-02  
适用范围：Qwen2 / NVIDIA 推理路径（Project #2 阶段）

---

## 1. 文档目的
本记录用于回答三件事：
1. 做过哪些优化，哪些保留，哪些回退。
2. 为什么做，怎么测，结果是否可信。
3. 当前性能位置和下一步方向。

说明：原始日志中存在重复条目、阶段重置和中间草稿。本版已归并为可追溯时间线，保留关键数据与结论。

---

## 2. 统一测试口径（当前生效）

### 2.1 基础命令
```bash
# 算子级
python test/ops/linear.py --device nvidia --profile
python test/ops/self_attention.py --device nvidia --profile

# 端到端（确定性）
python test/test_infer.py --model /home/wgreymon/model_pkg/DeepSeek-R1-Distill-Qwen-1.5B/ --device nvidia --test

# 端到端（性能）
python test/test_infer.py --model /home/wgreymon/model_pkg/DeepSeek-R1-Distill-Qwen-1.5B/ --device nvidia --max_steps 32
```

### 2.2 口径修复（关键）
`test/test_infer.py` 先跑 Torch 再跑 LLAISYS 时，已加入：
- `del model`
- `gc.collect()`
- `torch.cuda.empty_cache()`
- `torch.cuda.synchronize()`

意义：避免同进程中 Torch CUDA 缓存干扰 LLAISYS，防止出现“同命令一次 20s+、一次 1s+”的误判。

### 2.3 判定规则
1. 优先看同口径 A/B。
2. 正确性失败或崩溃，直接回退。
3. 无稳定收益（持平/退化/仅单次波动）回退。

---

## 3. 初始基线快照（2026-03-02）

| 场景 | Torch | LLAISYS | 结论 |
|---|---:|---:|---|
| linear f32 `(512,4096)x(4096,4096)` | 2.70780ms | 2.05755ms | LLAISYS 更快 |
| linear f16 `(512,4096)x(4096,4096)` | 0.60095ms | 0.58783ms | 接近持平 |
| linear bf16 `(512,4096)x(4096,4096)` | 0.55254ms | 0.58733ms | LLAISYS 略慢 |
| self_attention 小规模 case（f32/f16/bf16） | ~0.60ms | 0.03~0.06ms | LLAISYS 更快（小 shape） |
| `test_infer --test` | 通过 | 通过 | token 对齐 |

---

## 4. 优化时间线（合并去重）

### 阶段 A：热点定位与首轮实验（S001-S011）

| Step | 主要动作 | 关键结果 | 结论 |
|---|---|---|---|
| S001-S002 | 建立日志与基线 | 完成统一命令与初始测量 | 保留 |
| S003 | 减少 decode 冗余分配/无效开销 | `--max_steps 32: 9.28s -> 8.74s`；`--test: 24.49s -> 23.20s` | 有效，保留思想 |
| S004 | allocator 缓存池实验 | `8.74s -> 8.79s`，近似无收益 | 回退 |
| S005 | 引入 profile（`LLAISYS_PROFILE=1`） | layer 占比：`linear 94.525%`，`attn 0.651%` | 保留（定位能力） |
| S006 | QKV 融合 linear（decode） | `8.89s/8.91s`，较基线无优势 | 回退 |
| S006 补充 | 线性算子复测 | 单算子与 Torch 接近，但不足解释端到端差距 | 结论保留 |
| S007 | gate+up 融合 linear | `9.62s~9.70s`，明显变慢 | 回退 |
| S008 | 减少 host 侧 `slice` 开销 | `8.85s`，与基线持平/略慢 | 回退 |
| S009 | `M=1` fast path（sgemm->sgemv）实验 | 算子级接近，端到端无收益 | 回退 |
| S010 | decode 分阶段计时 | `forward 99.766%`，host/D2H 可忽略 | 保留（关键结论） |
| S011 | 无拷贝版 gate+up 融合重试 | `9.63s` 与 `10.76s`，显著退化 | 回退 |

阶段 A 结论：
1. decode 主要瓶颈在 GPU `forward`，不是 host 准备/D2H。
2. `linear` 是主热点，简单融合并未自动带来收益。
3. “减少调用数”必须结合 kernel 特性与中间数据流，不能只做结构级拼接。

### 阶段 B：稳定性排查与重置（S012-S015）

| Step | 主要动作 | 关键结果 | 结论 |
|---|---|---|---|
| S012 | `update_kv_cache` 改 async memcpy | 触发 `Segmentation fault` | 回退 |
| S013 | allocator 池化开关隔离 | 池化开/关都出现 `exit=139` | 崩溃非单一 allocator 原因 |
| S014 | 计划隔离 decode QKV 融合路径 | 在该轮工作树未形成稳定落地结果 | 历史记录保留，结论不纳入基线 |
| S015 | Lazy Allocation / 张量成员复用重试 | `8.9s -> 10.3s`，退化 | 回退 |

阶段 B 结论：
1. 不稳定实验必须先回退，再优化。
2. 单步实验与工作树一致性（代码/日志对应）要严格执行。

### 阶段 C：重置后的有效改进（S100-S105）

| Step | 主要动作 | 关键结果 | 结论 |
|---|---|---|---|
| S100 | 移除 zero-bias 路径的 dummy bias | `25.26s / 26.26s`，无明显收益 | 回退 |
| S101 | 小范围 `ensure_tensor` 复用 | `24.74s / 25.52s`，无稳定收益 | 回退 |
| S102 | 扩展到 layer 高频临时张量复用 | `24.81s -> 23.27s / 23.29s`，约 `6%` 提升 | 保留 |
| S200 | attention `seqlen=1` 快路径 | `24.15s / 25.33s`，波动并退化 | 回退 |
| S201 | 模型侧对象构造减法 | `27.28s` 且出现异常长跑 | 回退 |
| S103 | 基线确认（S102 状态） | `--max_steps 32: 9.98s / 9.86s` | 作为稳定基线 |
| S104 | KV 写回 async 再试 | `10.04s`，无改善 | 回退 |
| S105 | argmax 调度/内核实验 | `9.87s` 与 `11.89s`，波动大 | 回退 |

阶段 C 结论：
1. 当前真正稳定有效的代码级优化是 S102（高频张量复用）。
2. attention/argmax/KV 写回方向在现阶段都未形成稳定正收益。

### 阶段 D：测试体系完善与阶段验收（S106-S108）

| Step | 主要动作 | 关键结果 | 结论 |
|---|---|---|---|
| S106 | 修复 `test_infer` 同进程干扰 | 样本：LLAISYS `25.96s -> 1.64s`（口径修复后） | 保留，属于测试体系关键修复 |
| S107 | 新增 `test/benchmark_infer.py`（子进程隔离） | 支持多 prompt/多 token/多 backend、p50/p95/tok/s、hash 对比 | 保留 |
| S108 | 综合 benchmark 分析 | 9 个 case 中 8 个更快，平均时延改善约 `7.41%`，吞吐提升约 `8.08%` | 阶段性达成项目二性能目标 |

---

## 5. 当前保留项（代码与流程）

### 5.1 代码层
1. S102：decode 高频临时张量复用（`ensure_tensor` 扩展版）。
2. 采样链路已贯通（`top_k/top_p/temperature`），可用于项目三服务化与流式场景。

### 5.2 测试层
1. S106：`test/test_infer.py` 口径修复（Torch->LLAISYS 之间释放 CUDA 缓存）。
2. S107：`test/benchmark_infer.py` 作为统一综合对比入口。

---

## 6. 关键结论（截至 2026-03-02）

1. decode 端到端瓶颈明确在 GPU `forward`，host 与 D2H 占比很小。  
2. `linear` 是核心热点，但“简单融合”多次验证未形成稳定收益。  
3. 单算子接近 Torch 不等于端到端接近 Torch，decode 场景更受调度与整体执行路径影响。  
4. 在修正测试口径后，LLAISYS 已在多数真实 case 中达到与 Torch 同级或更优。  

---

## 7. 风险与待解释项

1. 确定性参数下存在 `medium/32` 单例 `output_match = N`，需要专项回归。  
2. `long/16` case 中 LLAISYS 略慢（`354.30ms` vs `340.39ms`），需观察是否为短输出开销主导。  
3. 历史实验中曾出现 `Segmentation fault`，后续涉及 async/memory 路径必须先做稳定性门禁。  

---

## 8. 下一阶段计划（建议）

1. 建立“确定性一致性回归”脚本：固定 `top_k=1, top_p=1, temperature=1`，批量校验 token 全量一致。  
2. 做 `lm_head(out_linear)` 的 decode 专项优化 A/B（重点 `M=1, N=vocab`）。  
3. 在稳定前提下评估 decode CUDA Graph，目标是降低小算子 launch 开销。  
4. 保留统一 benchmark 口径，所有优化只接受“3 次以上中位数稳定收益”。  

---

## 9. 后续记录模板

### SXXX
- 日期：
- 目标：
- 假设：
- 改动文件：
- 测试命令：
- 结果：
- 结论（保留/回退）：
- 下一步：

