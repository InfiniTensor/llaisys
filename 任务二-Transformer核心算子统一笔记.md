# 任务二统一笔记：RMSNorm、RoPE、Self-Attention、SwiGLU

## 0. 这份笔记解决什么问题

这是一份基于你当前实现代码整理的统一笔记，把四个模块放进同一条逻辑链：

**RMSNorm →（线性投影得 Q/K/V）→ RoPE(Q,K) → Self-Attention → FFN 中的 SwiGLU**。

目标是回答三个问题：
1. 每个模块在 Transformer Block 里负责什么；
2. 数学上怎么计算；
3. 在你的 C++ 实现里如何落地（形状、dtype、边界条件、数值稳定）。

---

## 1. 先看整体：一个 Transformer Block 的信息流

以 Pre-Norm 结构为例，可简化为：

1. 输入隐藏状态 `x`
2. `x1 = RMSNorm(x)`
3. `q,k,v = Linear(x1)`（不同权重）
4. `q,k = RoPE(q,k, pos_ids)`
5. `attn = SelfAttention(q,k,v)`（含 causal mask）
6. 残差连接
7. `x2 = RMSNorm(...)`
8. FFN 中 `out = SwiGLU(gate, up)`（再接线性投影回 `d_model`）
9. 残差连接输出

这四个模块分别承担：
- **RMSNorm**：稳住尺度，减少数值漂移；
- **RoPE**：把位置信息编码进 Q/K 的方向；
- **Self-Attention**：按相关性聚合上下文；
- **SwiGLU**：在 FFN 内动态筛选和增强特征。

---

## 2. RMSNorm：先把尺度稳定住

### 2.1 数学定义

对每一行向量 $x_i \in \mathbb{R}^d$：

$$
\text{rms}(x_i)=\sqrt{\frac{1}{d}\sum_{j=1}^{d}x_{i,j}^2+\epsilon}
$$

$$
y_{i,j}=w_j\cdot\frac{x_{i,j}}{\text{rms}(x_i)}
$$

### 2.2 直觉

- 不做去均值，只做“按 RMS 缩放”；
- 让不同 token 的向量幅度回到可控范围；
- 有助于深层网络训练/推理稳定。

### 2.3 你实现里的关键点

对应实现：`src/ops/rms_norm/op.cpp`

- 输入输出同 device、同 dtype；
- 要求 contiguous；
- `weight` 是 1D，长度等于最后一维；
- 计算时统一转 `float`（F16/BF16 避免精度问题）；
- 当前只支持 CPU；
- 支持 `F32/F16/BF16` 三种分发。

---

## 3. RoPE：把“位置”写进 Q/K 的方向

### 3.1 数学定义

对于每个位置 `pos_id` 和维度对 `j`：

$$
\phi_{j}=\frac{pos\_id}{\theta^{2j/d}}
$$

将向量按前后半维配对：$[x_j, x_{j+d/2}]$，做二维旋转：

$$
x'_j=x_j\cos\phi_j-x_{j+d/2}\sin\phi_j
$$

$$
x'_{j+d/2}=x_{j+d/2}\cos\phi_j+x_j\sin\phi_j
$$

### 3.2 为什么有效

- 旋转保持模长，主要改变方向；
- 通过不同频率（不同 `j`）编码多尺度位置信息；
- 天然适合相对位置关系建模。

### 3.3 你实现里的关键点

对应实现：`src/ops/rope/op.cpp`

- 输入形状按 3D 处理：`[seqlen, n(head), d]`；
- `pos_ids` 为 1D `int64`；
- 最后一维 `d` 必须是偶数；
- 前半维与后半维配对旋转；
- 计算角度时用 `pow/sin/cos`，当前 CPU 路径；
- dtype 分发支持 `F32/F16/BF16`。

---

## 4. Self-Attention：按相关性做上下文聚合

### 4.1 核心公式

$$
A=QK^T\cdot scale
$$

$$
P=\text{causal\_softmax}(A)
$$

$$
Y=PV
$$

其中 `scale` 常取 $1/\sqrt{d}$。

### 4.2 计算流程（与你实现一致）

对应实现：`src/ops/self_attention/op.cpp`

1. 对每个 `(token s, q-head h)` 计算与所有 key 位置的点积得分；
2. 乘 `scale`；
3. 加 causal mask（未来位置置 `-inf`）；
4. softmax（减最大值做数值稳定）；
5. 用权重对 `V` 做加权和得到输出。

### 4.3 形状与 GQA 约束

你的实现使用了 GQA 映射：

- `q`: `[seqlen, nhead, d]`
- `k`: `[total_len, nkvhead, d]`
- `v`: `[total_len, nkvhead, dv]`
- `out`: `[seqlen, nhead, dv]`
- 约束：`nhead % nkvhead == 0`

通过 `group_size = nhead / nkvhead`，把多个 Q 头映射到同一个 KV 头。

### 4.4 数值与工程细节

- softmax 前维护 `max_score`，降低溢出风险；
- `-inf` 位置在 exp 前置 0；
- 若 `sum_exp == 0`，回退为 1 防止 NaN；
- 内部累加用 `float`，最后 cast 回目标 dtype。

---

## 5. SwiGLU：FFN 里的动态门控激活

### 5.1 数学定义

逐元素计算：

$$
out_i=up_i\cdot\frac{gate_i}{1+e^{-gate_i}}
$$

等价于：

$$
out_i=up_i\cdot \text{SiLU}(gate_i)
$$

### 5.2 直觉

- `up` 分支提供“候选特征值”；
- `gate` 分支提供“通过强度”；
- 两者逐元素相乘，实现“按上下文动态开关特征”。

### 5.3 你实现里的关键点

对应实现：`src/ops/swiglu/op.cpp`

- 要求三个张量同形状、同 dtype、contiguous；
- 当前 CPU-only；
- 为数值稳定加了阈值分支：
  - `gate >= 50` 直接近似为 `gate`；
  - `gate <= -50` 近似为 `0`；
  - 中间区间走 `exp`；
- 支持 `F32/F16/BF16`。

---

## 6. 四个模块如何形成“闭环”

把它们串起来看，一个 token 的信息处理是：

1. **RMSNorm** 先统一尺度，避免后续点积过大/过小；
2. **RoPE** 把位置写入 Q/K 的方向，使注意力可感知相对位置；
3. **Self-Attention** 用 QK 相似度在历史上下文中检索并聚合 V；
4. **SwiGLU** 在 FFN 中再做一次非线性特征筛选与增强。

因此它们是互补关系：
- RMSNorm 解决“数值稳定”；
- RoPE 解决“位置信息”；
- Attention 解决“上下文路由”；
- SwiGLU 解决“特征表达能力”。

---

## 7. 与你任务二实现的对齐清单

- 统一支持 dtype：`F32/F16/BF16`（四个算子都做了分发）；
- 统一前置检查：device、dtype、contiguous、shape；
- 统一计算策略：内部尽量转 `float` 累加，再 cast 回目标类型；
- 统一设备边界：当前都以 CPU 实现为主；
- Attention 已覆盖 GQA + causal mask + 数值稳定 softmax。

---

## 8. 面试时可直接复述的总结

> 在任务二里，我把 Transformer 的核心算子链路打通了：
> 先用 RMSNorm 稳定激活尺度，再对 Q/K 做 RoPE 注入相对位置信息，然后用带 causal mask 的 self-attention 聚合上下文，最后在 FFN 里通过 SwiGLU 做动态门控非线性。工程上我统一做了 shape/dtype/device 校验，CPU 路径支持 F32/F16/BF16，核心计算用 float 累加保证稳定性。

这句话可以作为你“实现 + 理解”一体化回答的开场。
