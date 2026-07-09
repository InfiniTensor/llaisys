# 任务二统一笔记（完整版）：RMSNorm、RoPE、Self-Attention、SwiGLU

## 0. 笔记目标与阅读方式

这份笔记把你任务二实现的四个核心模块串成一条完整逻辑链：

**RMSNorm -> 线性投影(Q/K/V) -> RoPE(Q,K) -> Self-Attention -> FFN中的SwiGLU**。

建议按下面三层来读：
1. **先看“作用”**：每个算子解决什么问题；
2. **再看“数学”**：核心公式与直觉；
3. **最后看“工程”**：你代码里的输入约束、数值稳定、复杂度和常见坑。

---

## 1. Transformer Block 里的位置关系（总览）

以你当前实现对齐的 Pre-Norm 思路，可写成：

1. 输入隐藏状态 `x`
2. `x1 = RMSNorm(x)`
3. `q,k,v = Linear(x1)`
4. `q,k = RoPE(q,k,pos_ids)`
5. `attn = SelfAttention(q,k,v,scale)`（含 causal mask，支持 GQA）
6. 残差连接
7. `x2 = RMSNorm(...)`
8. FFN 内部 `ffn_mid = SwiGLU(gate,up)`，再线性投影回 `d_model`
9. 残差连接输出

四个模块的分工：
- **RMSNorm**：稳定向量尺度，减少训练/推理数值漂移；
- **RoPE**：将位置信息注入 Q/K 的方向中；
- **Self-Attention**：按相关性检索并聚合上下文；
- **SwiGLU**：在 FFN 中进行动态门控的非线性特征选择。

---

## 2. RMSNorm：先把输入尺度压稳

### 2.1 数学定义（纯文本）

对每一行向量 `x_i in R^d`：
- `rms(x_i) = sqrt((1/d) * sum_{j=1..d}(x_{i,j}^2) + eps)`
- `y_{i,j} = w_j * x_{i,j} / rms(x_i)`

### 2.2 为什么有效

- 不做去均值，只控制尺度；
- 对深层网络来说，尺度稳定通常比“强中心化”更关键；
- 相比 LayerNorm 少一部分统计量计算，工程上更轻量。

### 2.3 你实现里的落地点

对应：`src/ops/rms_norm/op.cpp`

- 校验：同 device、同 dtype、contiguous；
- 形状：`weight` 必须是 1D 且长度等于最后一维；
- 路径：CPU-only；
- dtype：`F32/F16/BF16` 分发；
- 数值：内部转 `float` 计算，再 cast 回目标类型。

### 2.4 实现步骤（工程视角）

1. 计算 `norm_dim = last_dim`，`outer_size = numel / norm_dim`；
2. 对每行先求平方和，再开方得到 `rms`；
3. 对每元素做 `x/rms * weight`；
4. 写回输出。

### 2.5 常见坑

- `eps` 太小会在极小输入上不稳定；
- 忘记 `weight` 与最后一维对齐；
- 直接用半精度累加易引入明显误差。

---

## 3. RoPE：将“位置”编码到方向

### 3.1 数学定义（纯文本）

对位置 `pos_id` 和维度对 `j`：
- `phi_j = pos_id / (theta^(2j/d))`

向量按前后半维配对：`[x_j, x_{j+d/2}]`，旋转为：
- `x'_j = x_j * cos(phi_j) - x_{j+d/2} * sin(phi_j)`
- `x'_{j+d/2} = x_{j+d/2} * cos(phi_j) + x_j * sin(phi_j)`

### 3.2 为什么有效

- 旋转保持模长，主要改变方向；
- 多频率（不同 `j`）编码多尺度位置关系；
- Attention 的内积对相对位移敏感，利于长上下文泛化。

### 3.3 你实现里的落地点

对应：`src/ops/rope/op.cpp`

- 输入按 3D：`[seqlen, nhead(or nkvhead), d]`；
- `pos_ids`：1D 且 `int64`；
- 约束：`d` 必须为偶数；
- 路径：CPU-only；
- dtype：`F32/F16/BF16` 分发。

### 3.4 三重循环结构（与你代码一致）

1. 遍历 `seq`（取 `pos_id`）；
2. 遍历 `head`；
3. 遍历 `j in [0, d/2)` 进行成对旋转。

### 3.5 常见坑

- `d` 不是偶数会直接破坏配对逻辑；
- `pos_ids` 类型错（非 int64）会导致位置读取错误；
- 索引展平时 base 偏移计算错误最常见。

---

## 4. Self-Attention：上下文检索与聚合核心

### 4.1 核心公式（纯文本）

- `A = QK^T * scale`
- `P = causal_softmax(A)`
- `Y = P * V`

其中 `scale` 常取 `1/sqrt(d)`，用于抑制点积随维度增长而放大。

### 4.2 你实现中的形状定义

对应：`src/ops/self_attention/op.cpp`

- `q: [seqlen, nhead, d]`
- `k: [total_len, nkvhead, d]`
- `v: [total_len, nkvhead, dv]`
- `out: [seqlen, nhead, dv]`

核心约束：
- `seqlen <= total_len`（支持 KV cache 场景）；
- `k.shape[2] == d`；
- `v.shape[0..1]` 与 `k.shape[0..1]` 对齐；
- `nhead % nkvhead == 0`（GQA）。

### 4.3 GQA 映射逻辑

- `group_size = nhead / nkvhead`
- `kv_head = q_head / group_size`

直觉：多个 Q 头共享较少的 KV 头，显著降低 KV cache 占用，同时尽量保留效果。

### 4.4 计算步骤（与你实现一致）

1. 逐 `(s,h)` 计算 `scores[t] = dot(Q[s,h], K[t,kv_head]) * scale`；
2. 根据因果可见范围做 mask（未来位置置 `-inf`）；
3. softmax（减最大值稳定化）；
4. 用 `scores` 加权求和 `V[:,kv_head,:]`，得到 `out[s,h,:]`。

### 4.5 数值稳定细节

- 先减 `max_score` 再 `exp`；
- `-inf` 分支先置 0；
- `sum_exp == 0` 时回退 1，避免 NaN；
- 累加统一用 `float`。

### 4.6 复杂度与性能认知

- 朴素实现主复杂度约为 `O(seqlen * nhead * total_len * d)`；
- 你这版更偏“正确性与清晰度优先”；
- 后续可优化方向：向量化（SIMD）、并行（OpenMP）、分块 softmax、KV cache 访问局部性优化。

---

## 5. SwiGLU：FFN 中的动态门控非线性

### 5.1 数学定义（纯文本）

- `out_i = up_i * gate_i / (1 + exp(-gate_i))`
- 等价写法：`out_i = up_i * SiLU(gate_i)`

### 5.2 机制直觉

- `up` 分支负责“给候选值”；
- `gate` 分支负责“给通过强度”；
- 二者逐元素相乘，实现对特征的动态筛选。

### 5.3 你实现里的落地点

对应：`src/ops/swiglu/op.cpp`

- 校验：同 device、同 dtype、同 shape、contiguous；
- 路径：CPU-only；
- dtype：`F32/F16/BF16`；
- 数值稳定阈值：
  - `gate >= 50` 近似 `gate`；
  - `gate <= -50` 近似 `0`；
  - 中间区间正常计算 `exp`。

### 5.4 为什么要做阈值分支

- 直接算 `exp(-gate)` 在极值时容易上溢/下溢；
- 阈值近似保留函数形状，同时消除不必要的数值风险。

### 5.5 常见坑

- 忘记形状一致性检查；
- 极值输入导致 `exp` 溢出；
- 半精度直接算导致误差放大。

---

## 6. 四模块闭环：为什么必须一起看

一个 token 在块内经历的是：

1. **RMSNorm**：先稳尺度，避免后续点积失控；
2. **RoPE**：把位置写入 Q/K 方向，构建相对位置信号；
3. **Self-Attention**：在历史上下文中检索并聚合信息；
4. **SwiGLU**：在 FFN 中对聚合后的特征做动态筛选与增强。

它们是互补关系，不是替代关系：
- RMSNorm 解决“稳定性”；
- RoPE 解决“位置建模”；
- Attention 解决“信息路由”；
- SwiGLU 解决“表达能力”。

---

## 7. 与你任务二代码的对齐清单（可直接用于答辩）

### 7.1 一致的工程规范

- 前置检查统一：`device / dtype / contiguous / shape`；
- 统一 dtype 分发：`F32/F16/BF16`；
- 统一数值策略：关键累加转 `float`；
- 统一设备边界：当前实现聚焦 CPU。

### 7.2 模块级亮点

- RMSNorm：按最后一维归一化，`weight` 对齐严格；
- RoPE：偶数维约束 + 前后半维配对旋转；
- Self-Attention：causal mask + 稳定 softmax + GQA 映射；
- SwiGLU：门控激活 + 极值阈值近似。

### 7.3 你当前版本的定位

这是一版“**先正确、再优化**”的实现：
- 优势：逻辑清晰、可验证、便于对齐 PyTorch；
- 后续：可在不改变接口的前提下继续做算子级性能优化。

---

## 8. 常见追问与简答模板

1. **为什么 RMSNorm 不减均值？**
  - 目标是先稳定尺度，实践中效果足够好且计算更轻量。

2. **为什么 RoPE 要偶数维？**
  - 旋转是二维配对操作，必须按 `d/2` 对元素进行映射。

3. **为什么 attention 要乘 `1/sqrt(d)`？**
  - 抑制点积尺度膨胀，避免 softmax 饱和导致梯度问题。

4. **GQA 为什么可行？**
  - KV 头共享减少内存和带宽压力，通常以很小精度代价换更大推理收益。

5. **SwiGLU 为什么要做阈值分支？**
  - 规避极值输入下 `exp` 数值不稳定，提升鲁棒性。

---

## 9. 面试30秒总结（可直接背）

在任务二中，我把 Transformer 核心算子链路打通了：
先用 RMSNorm 稳定激活尺度，再对 Q/K 做 RoPE 注入位置信息，之后用带 causal mask 和 GQA 的 self-attention 聚合上下文，最后在 FFN 中用 SwiGLU 做动态门控非线性。工程上我统一了 shape/dtype/device/contiguous 检查，CPU 路径支持 F32/F16/BF16，并通过 float 累加与稳定 softmax 处理保障数值稳定。
