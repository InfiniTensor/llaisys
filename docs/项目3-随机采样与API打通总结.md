# 项目 #3 随机采样与 API 打通 — 实现总结

本文档说明：为支持「构建 AI 聊天机器人」而实现的**随机采样算子**、**采样参数（Temperature / Top-K / Top-P）**以及**从 C 到 Python 的 API 打通**的完整流程、做了哪些修改、以及为什么这样做。

---

## 一、背景与目标

### 1.1 项目 #3 的要求

项目 #3 是「构建 AI 聊天机器人」。要实现能与用户实时对话的聊天机器人，需要：

- **更自然的回复**：不能总是选「概率最高的那个词」（argmax），否则生成会过于死板、重复。
- **随机采样**：按概率从候选词中抽样，使每次回复有一定随机性。
- **可调参数**：用 Temperature、Top-K、Top-P 控制随机程度和候选范围。

因此，需要：

1. 实现一个**随机采样算子**，支持 Temperature、Top-K、Top-P。
2. 在推理路径中**用该算子替代原来的 argmax**（在需要随机时）。
3. **打通 API**：从 Python 的 `generate(top_k, top_p, temperature, seed)` 一路传到 C++ 的采样算子。

### 1.2 原先的流程（仅 argmax）

在实现前，推理流程是：

```
Python: model.generate(inputs, max_new_tokens, top_k=1, top_p=0.8, temperature=0.8)
         ↓ 循环
         llaisysQwen2ModelInfer(model, token_ids, ntoken)   // 只有 3 个参数
         ↓
C++:    前向 → 得到 logits → argmax(logits) → 返回 next_token
```

- `top_k`、`top_p`、`temperature` 在 Python 有参数，但**没有传给 C**，C 侧只做 argmax，所以「未在 C 侧使用」。
- 要支持随机采样，就必须：**扩展 C 的 Infer 接口** → **在 C++ 里根据参数选择 sample 或 argmax** → **实现 sample 算子**。

---

## 二、整体流程（实现后）

实现后的数据流如下。

```
用户 / 测试脚本
    ↓
Python: model.generate(inputs, max_new_tokens, top_k=50, top_p=0.8, temperature=0.8, seed=0)
    ↓ 每个 token 循环
    llaisysQwen2ModelInfer(model, token_ids, ntoken, temperature, top_k, top_p, seed)
    ↓
C API (qwen2.h)
    ↓
qwen2.cc: llaisysQwen2ModelInfer(...)
    ├─ 前向：embed → layers → norm → linear → 得到 logits
    ├─ 取最后一个位置的 logits → last_logit_1d
    ├─ 判断：use_sampling = (temperature 有效 且 (top_k>1 或 0<top_p<1))
    │   ├─ 是 → 若在 GPU，先把 logits 拷到 CPU → 调用 ops::sample(last_logit, temperature, top_k, top_p, seed) → next_token
    │   └─ 否 → 调用 ops::argmax(...) → next_token（贪心）
    └─ 返回 next_token
    ↓
ops::sample (src/ops/sample/op.cpp)
    ├─ 将 logits 转为 float（支持 f32/f16/bf16）
    ├─ Temperature：若 ≤0 或极小 → 直接 argmax；否则 logits /= temperature
    ├─ Top-K：只保留 logit 最大的 k 个，其余置 -inf
    ├─ Softmax
    ├─ Top-P：按概率排序，只保留累积概率达到 p 的前缀，再归一化
    └─ 按概率多项式采样 → 写入 out_idx（一个 int64）
```

这样，单机、单次生成就具备「随机采样 + Temperature / Top-K / Top-P」能力，为后续聊天服务器和 UI 打基础。

---

## 三、做了哪些事（按模块）

### 3.1 新增：随机采样算子 `src/ops/sample/`

| 文件 | 作用 |
|------|------|
| `op.hpp` | 声明 `void sample(tensor_t out_idx, tensor_t logits, float temperature, int top_k, float top_p, uint64_t seed)` |
| `op.cpp` | CPU 实现：logits→float、temperature、top-k、softmax、top-p、多项式采样 |

**为什么要单独做一个 sample 算子？**

- 与现有算子风格一致（如 `argmax` 在 `src/ops/argmax/`），便于维护和后续加 GPU 实现。
- 输入是 logits（1D）、输出是一个 token 索引（int64），接口清晰；采样参数一起传入，避免在 qwen2 里写一坨采样逻辑。

**为什么先只做 CPU？**

- 采样本身是标量/小向量运算，CPU 实现即可用；GPU 上可后续再实现。
- 模型在 GPU 时，当前做法是：把最后一维 logits 拷到 CPU，在 CPU 上调用 sample，再把得到的索引返回，这样无需实现 GPU 版 sample 也能跑通。

**Temperature / Top-K / Top-P 在代码里怎么做的？**

- **Temperature**：先对 logits 做 `logits[i] /= temperature`。温度接近 0 时等价于放大最大值，softmax 后几乎变成 one-hot，再采样就近似 argmax；温度大则分布更平，更随机。
- **Top-K**：用 `std::nth_element` 找到第 k 大的阈值，把小于该阈值的 logit 置为 -inf，再 softmax 时这些位置概率为 0，相当于只从「概率最高的 k 个」里采样。
- **Top-P（nucleus）**：先 softmax 得到概率，按概率从高到低排序，取最小的前缀使得累积概率 ≥ p，其余位置置 0 再归一化，再从该子集里采样。

**为什么支持 f32/f16/bf16？**

- 与模型其它部分一致；内部统一转成 float 做 softmax 和采样，数值稳定且实现简单。

---

### 3.2 暴露 C 层算子接口

| 文件 | 修改内容 |
|------|----------|
| `include/llaisys/ops.h` | 声明 `void llaisysSample(out_idx, logits, temperature, top_k, top_p, seed)` |
| `src/llaisys/ops.cc` | 实现 `llaisysSample`，内部调用 `llaisys::ops::sample`，并把 `unsigned long long seed` 转成 `uint64_t` |

**为什么要单独一个 C 的 Sample API？**

- 与其它 op（如 `llaisysArgmax`）一致，方便 Python/其它语言通过 ctypes 调用；若以后有独立脚本只做「给一段 logits，采样一个 token」，可以直接用这个 API。

---

### 3.3 扩展 Qwen2 推理接口（C 与 C++）

| 文件 | 修改内容 |
|------|----------|
| `include/llaisys/models/qwen2.h` | `llaisysQwen2ModelInfer` 增加 4 个参数：`float temperature, int top_k, float top_p, unsigned long long seed` |
| `src/llaisys/qwen2.cc` | ① 函数签名增加上述 4 个参数；② 包含 `../ops/sample/op.hpp`；③ 在得到 `last_logit_1d` 后，根据 `temperature/top_k/top_p` 判断 `use_sampling`；④ 若 `use_sampling`：必要时把 logits 拷到 CPU，创建 CPU 上的 `out_idx`，调用 `ops::sample`，把结果拷回 `next_token`；⑤ 否则沿用原有 argmax 逻辑；⑥ 注释说明「何时采样、何时贪心」 |

**为什么在 Infer 里判断「采样 vs 贪心」？**

- 保持一个入口函数：`llaisysQwen2ModelInfer` 同时支持「贪心（测试/对齐）」和「随机采样（聊天）」。
- 规则简单：`temperature` 有效且（`top_k > 1` 或 `0 < top_p < 1`）时用采样，否则用 argmax；这样 Python 侧 `top_k=1` 或 `temperature≈0` 即退化为原来的贪心行为，兼容现有测试脚本（如 `--test`）。

**为什么 GPU 时要把 logits 拷到 CPU 再 sample？**

- 当前只实现了 CPU 版 sample；若模型在 GPU，logits 在显存，不能直接在 GPU 上调用现有 sample 实现。因此先 D2H 拷贝到 CPU 的临时 tensor，在 CPU 上 sample，再把得到的索引（一个 int64）返回，这样无需改 Python 接口即可支持 GPU 模型 + 随机采样。

---

### 3.4 Python 绑定与 generate 传参

| 文件 | 修改内容 |
|------|----------|
| `python/llaisys_py/libllaisys/ops.py` | 为 `llaisysSample` 声明 argtypes（out_idx, logits, c_float, c_int, c_float, c_ulonglong），并加入 `load_ops` |
| `python/llaisys_py/ops.py` | 增加 `Ops.sample(out_idx, logits, temperature=1.0, top_k=0, top_p=0.0, seed=0)`，内部调 `llaisysSample` |
| `python/llaisys_py/libllaisys/qwen2.py` | `llaisysQwen2ModelInfer` 的 argtypes 增加 `c_float, c_int, c_float, c_ulonglong`（temperature, top_k, top_p, seed） |
| `python/llaisys_py/models/qwen2.py` | ① `generate` 增加参数 `seed=0`；② 调用 `llaisysQwen2ModelInfer` 时传入 `c_float(temperature), c_int(top_k), c_float(top_p), c_ulonglong(seed)`；③ 文档字符串改为说明「temperature、top_k、top_p 会传入 C 侧」，并说明 seed 含义 |

**为什么要传 seed？**

- 可复现：同一 prompt、同一组参数下，相同 seed 得到相同序列，便于调试和测试。
- `seed=0` 表示「每次用随机设备」，不保证复现；非 0 则用该种子初始化 `std::mt19937`。

**为什么用 c_ulonglong 表示 seed？**

- C 侧用 `unsigned long long`，与 64 位种子一致；Python 侧用 `c_ulonglong` 和 `c_ulonglong(seed)` 与之对应，避免跨平台位数问题。

---

### 3.5 单元测试（可选）

| 文件 | 作用 |
|------|------|
| `test/ops/sample.py` | 不依赖完整模型：① 用 numpy 构造 logits，拷贝到 llaisys Tensor，调 `Ops.sample`，检查返回索引在 [0, voc) 内；② 固定 logits、极小 temperature，检查退化为 argmax（返回最大 logit 的下标）。若环境无 torch 或 DLL 问题，可单独用该脚本验证 sample 绑定与行为。 |

---

## 四、为什么这样设计（简要）

1. **算子独立（sample op）**：采样逻辑集中在一个 op 里，支持多种 dtype、Temperature/Top-K/Top-P，将来加 GPU 或其它采样方式（如 beam）只需改/加算子，不动 qwen2 前向大逻辑。
2. **一个 Infer 接口**：通过参数控制「采样 vs 贪心」，调用方简单；测试用 `top_k=1` 即与原来行为一致。
3. **参数从 Python 直通 C++**：`generate(...)` 的 `top_k/top_p/temperature/seed` 原样传到 C，C 侧真正使用，注释中「未在 C 侧使用」的问题被消除。
4. **GPU 兼容**：在未实现 GPU sample 前，用「logits 拷到 CPU → CPU sample」保证 GPU 模型也能做随机采样，为后续优化留空间。

---

## 五、涉及文件一览

| 类型 | 路径 |
|------|------|
| 新增 | `src/ops/sample/op.hpp`, `src/ops/sample/op.cpp` |
| 修改 | `include/llaisys/ops.h`, `include/llaisys/models/qwen2.h` |
| 修改 | `src/llaisys/ops.cc`, `src/llaisys/qwen2.cc` |
| 修改 | `python/llaisys_py/libllaisys/ops.py`, `python/llaisys_py/libllaisys/qwen2.py`, `python/llaisys_py/ops.py`, `python/llaisys_py/models/qwen2.py` |
| 可选 | `test/ops/sample.py` |

构建时 `xmake` 会扫描 `src/ops/*/op.cpp`，因此无需改 `xmake.lua`，sample 会自动参与编译。

---

## 六、如何运行并检查效果

在项目根目录（即包含 `xmake.lua` 的目录）下按顺序执行即可。

### 6.1 编译并安装到 Python 包

```bash
# 编译 C++ 与动态库
xmake build

# 将生成的 llaisys.dll（或 libllaisys.so）复制到 python/llaisys_py/libllaisys/
# 这样 Python 的 import llaisys_py 会用到刚编译的版本
xmake install
```

### 6.2 方式一：完整推理测试（推荐，需模型）

依赖：Python 环境已安装 `torch`、`transformers`、`huggingface_hub` 等（见项目 README）。

**不指定 `--model` 时**：会自动从 Hugging Face 下载 `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`（约数 GB），首次较慢。

```bash
# 进入项目根目录后，让 Python 找到本项目的 test 和 llaisys（若未 pip install -e python）
# Windows PowerShell 示例：
$env:PYTHONPATH = "python;test"
# Linux/macOS：
# export PYTHONPATH=python:test

# 1）贪心模式：与 PyTorch 对齐，用于验证实现正确性
python test/test_infer.py --test

# 若已有本地模型目录，可指定路径避免重复下载：
# python test/test_infer.py --model D:/models/DeepSeek-R1-Distill-Qwen-1.5B --test
```

- 通过条件：终端打印的「Your Result」与「Answer」的 **Tokens 序列完全一致**，且无 AssertionError。
- 含义：LLAISYS 在 `top_k=1, temperature=1.0` 下走 argmax 分支，与 PyTorch 一致。

**随机采样模式**：同一命令去掉 `--test`，会使用默认 `top_k=50, top_p=0.8, temperature=1.0`，走采样分支。

```bash
# 2）随机采样：每次运行结果一般不同
python test/test_infer.py

# 自定义 prompt、生成长度、采样参数示例：
# python test/test_infer.py --prompt "你好，请介绍一下自己" --max_steps 64 --temperature 0.8 --top_k 40 --top_p 0.9
```

- 检查方式：多运行几次，观察「Your Result」的文本是否有所变化；或修改 `test_infer.py` 里传给 `generate` 的 `seed` 固定，两次用同一 seed 应得到相同结果。

### 6.3 方式二：仅测 sample 算子（不跑完整模型）

不加载大模型，只验证「从 logits 按概率采样一个 token」的算子和 Python 绑定是否正确。

```bash
# 在项目根目录，确保使用本项目的 llaisys（例如设置 PYTHONPATH=python）
# Windows PowerShell：
$env:PYTHONPATH = "python"
python test/ops/sample.py
```

- 通过条件：终端输出 `Sample op tests passed!`，且无报错。
- 说明：该脚本会 `import llaisys_py`，若项目配置为通过 `llaisys` 包拉取 `models`，则会间接导入 `torch`；若本机 torch 有 DLL 等问题，可能在此报错，此时以方式一在能跑通的环境中验证即可。

### 6.4 可选：用 pip 安装本项目后再跑

若已在本机用 pip 安装过本项目的 Python 包，可直接用：

```bash
pip install -e python
xmake build
xmake install
python test/test_infer.py --test
```

这样无需每次设置 `PYTHONPATH`，`import llaisys_py` 会使用当前项目下的包和刚安装的 DLL。

### 6.5 小结

| 目的           | 命令示例 |
|----------------|----------|
| 验证贪心对齐   | `python test/test_infer.py --test`（可加 `--model <path>`） |
| 看随机采样效果 | `python test/test_infer.py`（可加 `--prompt`、`--temperature` 等） |
| 只测 sample  op | `PYTHONPATH=python python test/ops/sample.py` |

以上即为「随机采样 + Temperature / Top-K / Top-P + API 打通」的完整流程、实现内容、设计原因与运行检查方式。
