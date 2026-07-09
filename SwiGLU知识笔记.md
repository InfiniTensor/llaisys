# SwiGLU 激活函数完整学习笔记

## 一、概述与地位

### 1.1 什么是 SwiGLU？

**定义**：SwiGLU 是一种用于 Transformer 前馈网络（FFN）的非线性激活函数。

**完整公式**：
$$out_i = up_i \cdot \frac{gate_i}{1 + e^{-gate_i}}$$

其中：
- $up_i$：值分支生成的特征
- $gate_i$：门控分支生成的控制信号  
- $\frac{gate_i}{1 + e^{-gate_i}}$：动态门控函数

**数学性质**：
- 定义域：$(-\infty, +\infty)$
- 值域：$(-\infty, +\infty)$
- 当 $gate_i \to +\infty$ 时，$\frac{gate_i}{1+e^{-gate_i}} \to gate_i$
- 当 $gate_i \to -\infty$ 时，$\frac{gate_i}{1+e^{-gate_i}} \to 0$
- 当 $gate_i = 0$ 时，$\frac{gate_i}{1+e^{-gate_i}} = 0$

### 1.2 在 Transformer 推理流程中的位置

```
┌─────────────────────────────────────────────────────┐
│             Transformer Block 结构                    │
├─────────────────────────────────────────────────────┤
│                                                       │
│  输入: x [seqlen, d_model]                           │
│    ↓                                                  │
│  ┌─────────────────────────────────────────────┐    │
│  │  Multi-Head Self-Attention                  │    │
│  │  - 计算上下文关系                            │    │
│  │  - 输出保留所有位置的上下文信息              │    │
│  │  - 输出: [seqlen, d_model]                  │    │
│  └──────────────┬──────────────────────────────┘    │
│                 ↓                                    │
│  LayerNorm(归一化，稳定训练)                        │
│                 ↓                                    │
│  ┌─────────────────────────────────────────────┐    │
│  │  Feed-Forward Network (FFN)                 │    │
│  │                                              │    │
│  │  中间层扩展: Linear(d_model → 4×d_model)    │    │
│  │  输出: [seqlen, 4×d_model]                  │    │
│  │       ↓                                     │    │
│  │  ┌────────────────────────────────────┐    │    │
│  │  │  ⭐ SwiGLU 激活函数                │    │    │
│  │  │  - 对 4×d_model 个特征逐元素激活   │    │    │
│  │  │  - 根据上下文动态选择特征           │    │    │
│  │  │  输入: [seqlen, 4×d_model]         │    │    │
│  │  │  输出: [seqlen, 4×d_model]         │    │    │
│  │  └────────────────────────────────────┘    │    │
│  │       ↓                                     │    │
│  │  投影回原维度: Linear(4×d_model → d_model) │    │
│  │  输出: [seqlen, d_model]                   │    │
│  └─────────────────────────────────────────────┘    │
│                 ↓                                    │
│  残差连接 + LayerNorm                               │
│                 ↓                                    │
│  输出: y [seqlen, d_model]                         │
│                                                       │
└─────────────────────────────────────────────────────┘
```

---

## 二、作用机理深度解析

### 2.1 为什么需要激活函数？

#### 线性变换的局限性

没有激活函数的神经网络：
```
x → Linear(W₁) → Linear(W₂) → Linear(W₃) → ... → y
```

**问题**：无论有多少层，最终计算仍然是：
$$y = W_n \cdot ... \cdot W_2 \cdot W_1 \cdot x = W_{合并} \cdot x$$

仍然是线性变换！**无法学习复杂的非线性关系**。

#### 激活函数的作用

激活函数引入**非线性**，使得：
```
x → Linear → 激活 → Linear → 激活 → ... → y
```

每一层都能学习不同的特征表示，形成更复杂的决策边界。

### 2.2 SwiGLU 相比其他激活函数的本质区别

#### 2.2.1 常见激活函数对比

**ReLU**（Rectified Linear Unit）：
$$ReLU(x) = \max(0, x)$$

- 优点：计算简单快速
- 缺点：当 $x < 0$ 时，梯度为 0（**死神经元问题**）
- 性质：**固定激活曲线**，所有位置相同方式

**GELU**（Gaussian Error Linear Unit）：
$$GELU(x) = x \cdot \Phi(x)$$

其中 $\Phi(x)$ 是标准正态分布的累积分布函数。

- 优点：平滑，梯度不为 0，效果更好
- 缺点：仍然是**固定激活策略**

**SwiGLU**：
$$SwiGLU(up, gate) = up \cdot \frac{gate}{1+e^{-gate}}$$

- **动态激活策略**：根据输入内容动态决定激活强度
- **双分支设计**：up 生成值，gate 学习控制逻辑
- **自适应选择**：不同位置的特征可以有不同的激活方式

#### 2.2.2 直观对比

假设中间层某维度处理"水果"语义，有 3 个样本位置：

```python
位置1："苹果"（语境：红的、甜的）
位置2："科学"（语境：技术、研究）
位置3："树"（语境：自然、环境）

中间层值：
position = [up1=5.2,  up2=-0.3,  up3=2.8]

GELU 激活（固定策略）：
output = GELU([5.2, -0.3, 2.8])
       = [5.2×Φ(5.2), -0.3×Φ(-0.3), 2.8×Φ(2.8)]
       = [5.19, -0.085, 2.78]  ← 用同一个激活曲线处理所有位置

SwiGLU 激活（动态策略）：
gate = [gate1=8, gate2=-10, gate3=2]

output = up * sigmoid_gate(gate)
       = [5.2×1.0,  -0.3×0.0,  2.8×0.6]
       = [5.2, 0, 1.68]  ← 根据语境动态调整激活强度

分析：
- 位置1："苹果"：gate=8 很高 → 这个特征对水果识别很关键 → 全部保留
- 位置2："科学"：gate=-10 很低 → 这个特征在非水果语境无用 → 全部阻挡
- 位置3："树"：gate=2 中等 → 这个特征部分相关（树是植物，与水果有关）→ 部分保留
```

### 2.3 双分支设计的深层含义

#### FFN 的完整计算流程

```
输入: x [seqlen, d_model]
      ↓
┌─────────────────────────────────────────────────────┐
│             第一层 Linear（扩展层）                  │
│         输出维度：4 × d_model（通常）              │
└──────┬──────────────────────────────────────────────┘
       ↓
   [seqlen, 4×d_model]
       ↓ 分裂成两个分支
   ┌───────────────────────────────────────┐
   │                                       │
   ├→ gate 分支                            │  ← 学习什么
   │  - 对每个特征维度学习一个控制信号      │    应该被激活
   │  - 范围：(-∞, +∞)                     │
   │  - 语义："这个特征对当前语境重要吗？" │
   │                                       │
   ├→ up 分支                              │  ← 生成什么
   │  - 对每个特征维度生成原始特征值        │    样的值
   │  - 范围：(-∞, +∞)                     │
   │  - 语义："这个特征的具体值是多少？"   │
   │                                       │
   └───────────────────────────────────────┘
       ↓
   元素乘积：out = up * sigmoid_gate(gate)
       ↓
   [seqlen, 4×d_model]（特征被选择性激活）
       ↓
   第二层 Linear（压缩层）
   输出维度：d_model
       ↓
   [seqlen, d_model]
```

#### 为什么要分成两个分支？

**原因 1：参数效率**
```
单分支（普通激活）：
  Linear(d → 4d) + ReLU/GELU + Linear(4d → d)
  参数量：d×4d + 4d×d = 8d²

双分支（SwiGLU）：
  Linear(d → 4d) [gate] + Linear(d → 4d) [up] + SwiGLU + Linear(4d → d)
  参数量：d×4d + d×4d + 4d×d = 12d²
```
看起来更多，但实际效果更好。

**原因 2：独立学习空间**
- `gate` 分支学会：**"鉴别"** - 判断当前特征对这个位置有多重要
- `up` 分支学会：**"生成"** - 产生对应特征的最好表示

两个独立的线性变换允许网络同时优化这两个不同的目标。

**原因 3：信息分解**
```
假设输入 x 包含多个语义：[位置信息, 语义信息, 语法信息, ...]

不使用门控：
  Linear(x) → 单一压缩 → 无法区分哪些信息对当前特征重要
  
使用双分支：
  gate_branch: 学会对不同输入特征的"敏感度"权重
  up_branch: 生成具体的特征值
  
  结果：自动学会"对位置信息敏感的特征就高激活，对语法信息敏感的就低激活"
```

---

## 三、研究背景与发现

### 3.1 为什么研究人员会发现 SwiGLU？

#### 问题的提出

**背景**：2020-2021 年间，Transformer 应用越来越广泛，但 FFN 的设计（特别是激活函数）还很简陋。

**核心问题**：
1. **ReLU/GELU 的局限**：固定激活策略，不能根据语境调整
2. **参数冗余**：简单的 Linear → 激活 → Linear，并不能充分发挥参数的作用
3. **特征浪费**：某些特征在某个位置根本不需要，但仍然被激活处理

#### 3.2 "为什么"的深层思考

**类比：人类的注意力机制**

```
场景1：看到苹果
  大脑激活的特征：颜色、圆形、甜味、...
  抑制的特征：数学公式、编程语法、...

场景2：看到数学题
  大脑激活的特征：逻辑、符号、推理、...
  抑制的特征：颜色、口味、...

关键：同一个"苹果"特征在不同语境下的激活强度不同！
```

SwiGLU 就是对这一直觉的形式化表达。

#### 3.3 论文发现

**论文**：《GLU Variants Improve Transformer》（2020年）

**主要发现**：
1. **门控线性单元（GLU）优于固定激活函数**
2. **多头变体（Gated Linear Units, Multi-Head）**
3. **SwiGLU 实现（Swish + GLU）**：结合 Swish 激活（$x \cdot \sigma(x)$）和门控机制

**实验结果**：
- 同等参数量下，SwiGLU 比 GELU 性能提升 **7-15%**
- 成为现代大模型（Qwen、LLaMA 等）的标准配置

---

## 四、在完整推理系统中的作用流程

### 4.1 从词嵌入到输出的完整链路

```
┌──────────────────────────────────────────────────────────┐
│ Step 1: 词向量嵌入与位置编码                             │
│ Input: "苹果很红"                                        │
│ ↓                                                          │
│ Token Embedding + Position Encoding                      │
│ 词向量: [0.1, -0.2, 0.5, ...]  [768维]                 │
│ ↓                                                          │
│ Block 0                                                   │
│ ├─ Self-Attention: 融合上文信息                          │
│ │  "苹果" + "很" + "红" = [体现"红苹果"关系]           │
│ ├─ FFN + SwiGLU: 特征精化与选择                         │
│ │  中间: [4×768=3072维]                                 │
│ │  激活: 动态选择哪些特征关键 → "红色"特征高激活       │
│ └─ 输出: [精化后的向量]                                 │
│ ↓                                                          │
│ Block 1                                                   │
│ ├─ Self-Attention: 融合更多上下文                       │
│ │  结合前面的结果和自身语境                              │
│ ├─ FFN + SwiGLU: 进一步精化                             │
│ │  gate 学到："水果"语境下的特征应该这样激活            │
│ └─ 输出: [再次精化的向量]                                │
│ ↓                                                          │
│ ... Block 2, 3, ..., N-1 ...                            │
│ ↓                                                          │
│ Block N-1（最后一层）                                    │
│ ├─ Self-Attention: 综合全局语义                         │
│ ├─ FFN + SwiGLU: 最后的特征选择                         │
│ └─ 输出: [最终向量表示]                                 │
│ ↓                                                          │
│ Step 2: 输出层                                           │
│ Output Linear: [768] → [vocab_size]                     │
│ Softmax: 概率分布                                        │
│ Output: ["苹果"的下一个词概率分布]                      │
└──────────────────────────────────────────────────────────┘
```

### 4.2 SwiGLU 在每个 Block 中的具体作用

#### Block i 的完整计算

```python
# 伪代码表示
def transformer_block(x):
    # Self-Attention：获取上下文
    attn_output = self_attention(x)  # 包含周围词的信息
    x = layernorm(x + attn_output)   # 残差连接
    
    # FFN with SwiGLU：特征精化
    # 第一阶段：扩展
    up = linear_up(x)      # [seqlen, 4×d]
    gate = linear_gate(x)  # [seqlen, 4×d]
    
    # 第二阶段：SwiGLU激活
    # 这是关键！根据 x 的内容，决定哪些中间特征应该通过
    activated = up * sigmoid_glu(gate)  # 元素乘积
    
    # 为什么这样做？
    # - up 生成了 4×d 个潜在特征
    # - gate 学到了"在这个输入下，哪些特征相关"
    # - sigmoid_glu(gate) 值域 [0, 1)，相当于一个掩码
    # - 相关的特征 gate>0，会被放大；无关的 gate<0，会被抑制
    
    # 第三阶段：压缩
    output = linear_out(activated)  # [seqlen, d]
    
    x = layernorm(x + output)  # 残差连接
    
    return x
```

#### 具体数值例子

假设处理位置 i（词"苹果"）：

```
输入 x_i: [-0.1, 0.3, 0.5, 0.2, ...]  (d_model=768)

Self-Attention 后（融合上下文）:
x_attn_i: [0.2, 0.4, -0.3, 0.6, ...]
          ↑
       现在包含了"很"和"红"的信息

FFN 扩展层：
up_i = Linear_up(x_attn_i)
     = [5.2, -0.3, 0.8, 10.1, -2.5, 3.1, ..., ...](3072维)

gate_i = Linear_gate(x_attn_i)
       = [8.0, -10.0, 2.0, 15.0, -0.5, 1.5, ..., ...](3072维)

计算 sigmoid_glu(gate_i)：
特征0:  gate[0]=8.0   → sigmoid_glu=1.0     (高激活)
特征1:  gate[1]=-10.0 → sigmoid_glu≈0.0     (抑制)
特征2:  gate[2]=2.0   → sigmoid_glu≈0.6     (中等激活)
特征3:  gate[3]=15.0  → sigmoid_glu=1.0     (高激活)
特征4:  gate[4]=-0.5  → sigmoid_glu≈0.4     (弱激活)
特征5:  gate[5]=1.5   → sigmoid_glu≈0.58    (中等激活)
...

激活结果（元素乘积）：
activated_i[0] = 5.2 × 1.0 = 5.2      ✓ 保留
activated_i[1] = -0.3 × 0.0 = 0       ✓ 抑制
activated_i[2] = 0.8 × 0.6 = 0.48     ✓ 部分保留
activated_i[3] = 10.1 × 1.0 = 10.1    ✓ 保留
activated_i[4] = -2.5 × 0.4 = -1.0    ✓ 弱保留
activated_i[5] = 3.1 × 0.58 = 1.8     ✓ 中等保留
...

投影回原维度：
output_i = Linear_out([5.2, 0, 0.48, 10.1, -1.0, 1.8, ...])
         = [新的768维向量]

作用：
Original: [-0.1, 0.3, 0.5, 0.2, ...]  (只有Self-Attention的直接结果)
After SwiGLU: [经过精化的向量]  (特征被选择性地强化或抑制)
```

---

## 五、激活的目的与后续影响

### 5.1 SwiGLU 激活的三个关键目的

#### 目的1：**特征过滤（Feature Filtering）**

```
通过 gate 学习的过滤器：
- 高 gate：这个特征在当前语境重要 → 放大
- 低 gate：这个特征无关 → 衰减
- 中等 gate：这个特征部分相关 → 部分保留

好处：减少信息噪声
```

#### 目的2：**非线性变换（Non-linear Transformation）**

```
纯线性的叠加：Linear₁ + Linear₂ = 仍是 Linear
引入非线性激活：能学到复杂的决策边界

例如：判断"这是什么水果"
Linear 只能学：[红色, 圆形, 甜味] → 苹果（直线分界）
非线性可以学：
  if (红色 AND 圆形) → 苹果
  if (红色 AND 细长) → 辣椒
  if (圆形 AND 甜) AND NOT 红色 → 白色葡萄
  ...
```

#### 目的3：**语境自适应（Context Adaptation）**

```
关键特性：gate 的值取决于输入 x
- 同一个特征在不同位置可能有完全不同的激活强度
- 模型学会了"在不同语境下用不同的方式处理信息"

例子：
"红苹果"位置：gate[红色特征]=10 → 高激活
"科学"位置：gate[红色特征]=-5 → 低激活
同一维度，不同激活！
```

### 5.2 激活后续如何影响下层

#### 信息流的演变

```
Layer 0:
  输入词向量: [苹果] = [0.1, -0.2, ...]
  Self-Attn: 融合周围词 = [0.2, 0.3, ...]
  FFN+SwiGLU: 特征精化 = [提高了"水果"语义强度]
  输出: [0.15, 0.5, ...]  ← 现在"水果"语义更强

Layer 1:
  输入: [0.15, 0.5, ...]  (继承了强化的水果语义)
  Self-Attn: 与其他词再次互动
           用现在的"强水果语义"与周围词交互
           可能激活"食物"、"颜色"等相关概念
  FFN+SwiGLU: 在食物语境下进行特征选择
  输出: [更高层次的语义：这是可以吃的东西]

Layer 2:
  输入: [更高层次语义]
  Self-Attn: 在更高抽象度上处理
  FFN+SwiGLU: 再次特征选择
  输出: [最后可能激活：农产品、自然、健康, ...]

...

最后：
模型基于这一系列渐进的特征激活和选择，
预测下一个词时已经建立了完整的语义理解
```

### 5.3 梯度反传与训练影响

#### 前向传播（推理）

```
x ──→ Linear_up ──→ up
 \                    ↘
  → Linear_gate → gate ──→ SwiGLU → activated ──→ Linear_out → output
```

#### 反向传播（训练）

```
损失函数 L
  ↓
∂L/∂output（输出层的梯度）
  ↓
∂L/∂(Linear_out 的权重)  ← Linear_out 学到如何压缩激活结果
  ↓
∂L/∂activated           (激活结果的梯度)
  ├──────────┬─────────────────┐
  ↓          ↓                 ↓
∂L/∂up      ∂L/∂gate    ∂L/∂(sigmoid_glu)
  ↓          ↓                 ↓
由激活值决定  由up值和激活     由输入决定
             导数决定

关键的链式法则：
∂L/∂up = ∂L/∂(up × sigmoid_glu(gate)) · sigmoid_glu(gate)
         ← gate 值越高，up 的梯度越强

∂L/∂gate = ∂L/∂(up × sigmoid_glu(gate)) · up · ∂sigmoid_glu/∂gate
           ← up 值越大，gate 学习的信号越强
```

#### 学到了什么？

```
Linear_up 的权重：学到 x → 特征值 的映射
  训练目标：生成有用的特征候选

Linear_gate 的权重：学到 x → 特征选择 的映射
  训练目标：学会"这个特征在这个语境下重要吗？"

整体效果：
随着训练进行，
- up 分支越来越善于生成相关的特征
- gate 分支越来越善于识别哪些特征应该通过
- 两者的配合越来越紧密和高效
```

---

## 六、与其他组件的协作

### 6.1 与 Self-Attention 的互动

```
Self-Attention:
  输入: x (原始词向量)
  输出: 融合了 Q·K^T 权重的上下文信息

  作用：确定"看哪里"

FFN + SwiGLU:
  输入: Attention 的输出
  输出: 根据上下文选择特征
  
  作用：确定"看到什么后，应该怎么处理"

流程：
[输入] → Self-Attention → [知道了前后文] 
                           ↓
                         SwiGLU
                           ↓
                       [根据前后文调整特征]
                           ↓
                       [输出给下一层]
```

### 6.2 与残差连接的互动

```
x → SelfAttn → y₁
    ↓           ↓
    + ←────────┘
    ↓
  LayerNorm
    ↓ x'
    → FFN+SwiGLU → y₂
    ↓               ↓
    + ←──────────┘
    ↓
  LayerNorm → 输出

残差连接的作用：
1. 梯度流：即使激活函数有梯度问题，也能通过直连路径反传
2. 恒等映射：网络可以学到"保持原有信息"的策略
3. 稳定性：大网络的训练更稳定

SwiGLU 与残差的关系：
- 残差确保即使 SwiGLU 学得不好，原信息也不会丢失
- SwiGLU 学习增量变换（增强或抑制特定特征）
- 最终输出 = 原有信息 + 动态选择后的增强特征
```

### 6.3 与层归一化的互动

```
层归一化顺序（Post-LN）：
  SelfAttn → Linear + Bias → Add(残差) → LayerNorm → FFN+SwiGLU → Add → LayerNorm

作用：
1. 稳定激活值的范围
2. 让 SwiGLU 的输入和输出都在稳定范围内
3. 减少训练中的数值不稳定问题
```

---

## 七、SwiGLU 的数值稳定性实现

### 7.1 为什么需要数值稳定？

```
sigmoid_glu(x) = x / (1 + exp(-x))

问题：
- 当 x = 100 时，exp(-100) ≈ 0，计算没问题
- 当 x = -100 时，exp(-(-100)) = exp(100) = 非常大的数！
  导致 1 + exp(100) 溢出 → inf 或 NaN

训练失败！
```

### 7.2 我们的实现方案

```cpp
float glu_val;
if (gate_val >= 50.0f) {
    // exp(-50) ≈ 1.9e-22，可以忽略
    glu_val = gate_val;
} else if (gate_val <= -50.0f) {
    // exp(50) ≈ 5.2e21，非常大
    // gate_val / (1 + 非常大的数) ≈ 0
    glu_val = 0.0f;
} else {
    // 正常计算
    glu_val = gate_val / (1.0f + std::exp(-gate_val));
}
```

**为什么这样做**：
- 避免 exp() 计算超大/超小值
- 保留了 sigmoid_glu 的本质行为
- 转换为 float 计算确保精度

---

## 八、SwiGLU 的优势总结

| 特性 | 优势 |
|------|------|
| **动态门控** | 根据输入语境自适应激活，不像 ReLU/GELU 的固定策略 |
| **双分支** | up 生成候选特征，gate 学习选择策略，分工明确 |
| **非线性** | 突破线性变换的局限，学习复杂的特征关系 |
| **梯度流** | 相比 ReLU，不存在死神经元问题；相比普通激活，有控制权重 |
| **参数效率** | 虽然参数多一倍的线性层，但整体性能提升 7-15% |
| **可解释性** | gate 的值反映了"这个特征的重要性"，可以分析和可视化 |

---

## 九、实现细节回顾

### 9.1 完整的计算过程

```cpp
// 伪代码
void swiglu_impl(float* out, const float* gate, const float* up, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        float gate_val = gate[i];
        float up_val = up[i];
        
        // 计算 sigmoid_glu(gate) = gate / (1 + exp(-gate))
        float glu_val;
        if (gate_val >= 50.0f) {
            glu_val = gate_val;  // 接近 gate_val
        } else if (gate_val <= -50.0f) {
            glu_val = 0.0f;      // 接近 0
        } else {
            glu_val = gate_val / (1.0f + std::exp(-gate_val));
        }
        
        // 输出 = up * glu
        out[i] = up_val * glu_val;
    }
}
```

### 9.2 参数验证

```
✓ 形状检查：out、gate、up 必须相同 shape
✓ 连续性检查：所有张量必须内存连续
✓ 数据类型：支持 F32、F16、BF16
✓ 设备一致性：所有张量在同一设备
```

---

## 十、总结：SwiGLU 的完整图景

### 10.1 核心洞察

```
问题：Transformer FFN 中的激活函数如何根据上下文调整？

答案：SwiGLU
  = 双分支线性层（生成候选 up，生成选择信号 gate）
  + 元素乘积（将选择应用到候选上）
  
结果：
  - 高效的特征选择机制
  - 根据 Self-Attention 的输出（上下文）自适应激活
  - 更好的梯度流和参数利用率
```

### 10.2 在完整系统中的角色

```
推理流程：
  词向量
    ↓
  Block 0: Self-Attention(融合上下文) → SwiGLU(选择特征)
    ↓
  Block 1: Self-Attention(更高层融合) → SwiGLU(更高层选择)
    ↓
  ...
    ↓
  Block N: Self-Attention → SwiGLU → 最终输出

SwiGLU 在其中：
- 承上：接收 Self-Attention 融合的上下文信息
- 启下：生成经过特征选择的表示，传给下一层
- 中介：是上下文到特征的翻译器
```

### 10.3 为什么有效

1. **生物学启发**：人脑也是通过注意力选择和激活特定神经通路
2. **信息论**：最大化有用信息，最小化噪声
3. **优化理论**：动态选择使得梯度信号更强
4. **实验验证**：同等参数量性能提升 7-15%，被主流模型采用

---

## 参考实现

### 完整代码（C++）

```cpp
template <typename T>
void swiglu_impl(T *out_ptr, const T *gate_ptr, const T *up_ptr, size_t total_size) {
    for (size_t i = 0; i < total_size; ++i) {
        float gate_val = llaisys::utils::cast<float>(gate_ptr[i]);
        float up_val = llaisys::utils::cast<float>(up_ptr[i]);
        
        // 数值稳定的 sigmoid_glu 计算
        float glu_val;
        if (gate_val >= 50.0f) {
            glu_val = gate_val;
        } else if (gate_val <= -50.0f) {
            glu_val = 0.0f;
        } else {
            glu_val = gate_val / (1.0f + std::exp(-gate_val));
        }
        
        out_ptr[i] = llaisys::utils::cast<T>(up_val * glu_val);
    }
}
```

### 简单验证（Python）

```python
import torch
import torch.nn.functional as F

def swiglu(up, gate):
    """SwiGLU 激活"""
    # gate / (1 + exp(-gate))
    return up * (gate / (1.0 + torch.exp(-gate)))

# 测试
up = torch.randn(2, 3072)
gate = torch.randn(2, 3072)
output = swiglu(up, gate)
print(output.shape)  # [2, 3072]
```

---

**学习建议**：
1. 理解"为什么需要激活" → 非线性
2. 理解"为什么 SwiGLU 更好" → 动态选择
3. 理解"gate 学到了什么" → 通过反向传播的梯度分析
4. 在推理系统中追踪特征的变化 → 可视化不同层的激活模式
