## 项目 #2：GPU 集成

### 1. 架构设计

本次实现没有改动 LLAISYS 的整体执行框架，只在现有 `device -> ops -> model` 链路中插入 GPU 后端。

```text
Python API / Test
        |
        v
  LLAISYS C API
        |
        v
  Runtime / Tensor / Model
        |
        +-------------------+-------------------+
        |                   |                   |
        v                   v                   v
       CPU              NVIDIA GPU          MetaX GPU
                           |                   |
                           v                   v
               src/device/nvidia/     src/device/metax/
                           |                   |
                           +---------+---------+
                                     |
                                     v
                           算子分发 src/ops/<op>/op.cpp
                                     |
                   +-----------------+-----------------+
                   |                                   |
                   v                                   v
        src/ops/<op>/nvidia/*.cu          src/ops/<op>/metax/*.maca
                                                    |
                                                    v
                                      复用 ../nvidia/*.cu 算子主体
```

- 设备层：
  - `src/device/nvidia/` 实现 NVIDIA Runtime API 与设备资源管理
  - `src/device/metax/` 实现 MetaX Runtime API 入口
- 算子层：
  - `src/ops/*/nvidia/` 实现 CUDA 算子
  - `src/ops/*/metax/` 作为 MetaX 编译入口
- 构建层：
  - `xmake/nvidia.lua` 管理 CUDA/NVCC 编译
  - `xmake/metax.lua` 管理 MACA/MXCC 编译

核心设计是“平台分离、算子复用”：

- NVIDIA 路径使用原生 CUDA 构建与 runtime
- MetaX 路径单独提供设备枚举、构建规则和 runtime 分发
- MetaX 不重写整套算子，而是通过 `.maca` 入口复用 `nvidia/*.cu` 中的 CUDA-like 算子主体

因此，框架层面是两条独立 GPU 后端；算子源码层面只维护一套主实现。

### 2. 实现步骤

#### 2.1 NVIDIA 后端

第一步是补全 NVIDIA Runtime API，对齐 CPU Runtime 接口，包括：

- device count / set device
- malloc / free
- memcpy
- synchronize

随后在 `src/device/runtime_api.cpp` 中注册 NVIDIA runtime，使上层 `Tensor`、`RuntimeAPI` 和模型代码可以直接使用 GPU 设备。

第二步是接入 CUDA 构建链：

- 在 `xmake/nvidia.lua` 中加入 `.cu` 编译与链接规则
- 通过 `--nv-gpu=y` 控制是否启用 GPU 编译

第三步是补全 CUDA 算子。实现上采用统一模式：

- 每个算子在 `src/ops/<op>/nvidia/` 中提供 host 入口
- host 入口完成 dtype 分派、launch 配置与错误检查
- 计算逻辑写在模板化 kernel 中

实现重点在两个热点算子：

- `linear`
  - 采用“一线程对应一个输出元素”的映射
  - `fp16/bf16` 先转 `float` 再累加
- `self_attention`
  - 采用二维 grid，按 `(query, head)` 映射 block
  - 在 block 内完成 score 计算、softmax 和 value 加权
  - `scores` 使用 shared memory 存储

其余算子如 `add`、`rope`、`rms_norm`、`swiglu`、`embedding`、`argmax`、`rearrange` 按相同方式补齐，形成完整推理执行链。

#### 2.2 MetaX 后端

MetaX 的实现重点不在重新设计算子，而在接入新的设备路径。

实现步骤如下：

1. 新增 `ENABLE_METAX_API`
2. 新增 `LLAISYS_DEVICE_METAX` 与 Python 侧 `DeviceType.METAX`
3. 在 `runtime_api.cpp` 中加入 MetaX runtime 分发
4. 新增 `xmake/metax.lua`，使用 `mxcc` 编译 `.maca`
5. 为每个算子添加 `src/ops/*/metax/*.maca` 入口
6. 在 `.maca` 中复用 `../nvidia/*.cu` 算子主体

这样实现后，MetaX 具备独立设备语义，但不引入第二套重复算子实现。  
这一点是本次适配的关键取舍。

### 3. 测试

测试分两层进行。

#### 3.1 单算子测试

先逐个验证 GPU 算子：

```bash
python test/test_runtime.py --device nvidia
python test/ops/add.py --device nvidia
python test/ops/argmax.py --device nvidia
python test/ops/embedding.py --device nvidia
python test/ops/linear.py --device nvidia
python test/ops/rms_norm.py --device nvidia
python test/ops/rope.py --device nvidia
python test/ops/self_attention.py --device nvidia
python test/ops/swiglu.py --device nvidia
```

MetaX 路径使用同样方法，设备改为 `metax`。  
这样可以先验证 Runtime、dtype 分派和单算子正确性，再进入整模型测试。

#### 3.2 端到端推理测试

最终使用 `test/test_infer.py --test` 验证整条执行链。判断标准不是只看程序是否运行，而是：

- 生成 token 是否与参考一致
- 文本输出是否一致
- 测试是否通过

Nvidia推理测试结果如下：
```
(base) machine@dsw-607126-85f54bdf75-5lzlx:~/llaisys$ python test/test_infer.py --model ../models/DeepSeek-R1-Distill-Qwen-1.5B/ --test --device nvidia
`torch_dtype` is deprecated! Use `dtype` instead!
Loading model from local path: ../models/DeepSeek-R1-Distill-Qwen-1.5B/
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 339/339 [00:03<00:00, 95.80it/s]
The module name  (originally ) is not a valid Python identifier. Please rename the original module to avoid import issues.
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:151643 for open-end generation.
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.

=== Answer ===

Tokens:
[151646, 151644, 15191, 525, 498, 30, 151645, 151648, 198, 91786, 0, 358, 2776, 18183, 39350, 10911, 16, 11, 458, 20443, 11229, 17847, 3465, 553, 18183, 39350, 13, 358, 2776, 518, 697, 2473, 323, 1035, 387, 33972, 311, 7789, 498, 448, 894, 43883, 476, 9079, 498, 1231, 614, 624, 151649, 271, 91786, 0, 358, 2776, 18183, 39350, 10911, 16, 11, 458, 20443, 11229, 17847, 3465, 553, 18183, 39350, 13, 358, 2776, 518, 697, 2473, 323, 1035, 387, 33972, 311, 7789, 498, 448, 894, 43883, 476, 9079, 498, 1231, 614, 13, 151643]

Contents:
<｜User｜>Who are you?<｜Assistant｜><think>
Greetings! I'm DeepSeek-R1, an artificial intelligence assistant created by DeepSeek. I'm at your service and would be delighted to assist you with any inquiries or tasks you may have.
</think>

Greetings! I'm DeepSeek-R1, an artificial intelligence assistant created by DeepSeek. I'm at your service and would be delighted to assist you with any inquiries or tasks you may have.


Time elapsed: 9.36s


=== Your Result ===

Tokens:
[151646, 151644, 15191, 525, 498, 30, 151645, 151648, 198, 91786, 0, 358, 2776, 18183, 39350, 10911, 16, 11, 458, 20443, 11229, 17847, 3465, 553, 18183, 39350, 13, 358, 2776, 518, 697, 2473, 323, 1035, 387, 33972, 311, 7789, 498, 448, 894, 43883, 476, 9079, 498, 1231, 614, 624, 151649, 271, 91786, 0, 358, 2776, 18183, 39350, 10911, 16, 11, 458, 20443, 11229, 17847, 3465, 553, 18183, 39350, 13, 358, 2776, 518, 697, 2473, 323, 1035, 387, 33972, 311, 7789, 498, 448, 894, 43883, 476, 9079, 498, 1231, 614, 13, 151643]

Contents:
<｜User｜>Who are you?<｜Assistant｜><think>
Greetings! I'm DeepSeek-R1, an artificial intelligence assistant created by DeepSeek. I'm at your service and would be delighted to assist you with any inquiries or tasks you may have.
</think>

Greetings! I'm DeepSeek-R1, an artificial intelligence assistant created by DeepSeek. I'm at your service and would be delighted to assist you with any inquiries or tasks you may have.


Time elapsed: 83.64s

Test passed!
```

曦云 C500推理结果如下：
```
(base) root@d3871d5ad673:/home/machine/llaisys# python test/test_infer.py --test --device metax 
Loading model from Hugging Face: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
Fetching 9 files: 100%|████████████████████████████████████████████████████████████████████████████████| 9/9 [00:00<00:00, 102023.61it/s]
`torch_dtype` is deprecated! Use `dtype` instead!
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:151643 for open-end generation.
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
/opt/conda/lib/python3.10/site-packages/torch/nn/functional.py:5912: UserWarning: 1Torch was not compiled with memory efficient attention. (Triggered internally at /workspace/framework/mcPytorch/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp:649.)
  return _scaled_dot_product_attention(query, key, value, attn_mask, dropout_p, is_causal, scale = scale, enable_gqa = enable_gqa)

=== Answer ===

Tokens:
[151646, 151646, 151644, 15191, 525, 498, 30, 151645, 151648, 198, 91786, 0, 358, 2776, 18183, 39350, 10911, 16, 11, 458, 20443, 11229, 17847, 3465, 553, 18183, 39350, 13, 358, 2776, 518, 697, 2473, 323, 1035, 387, 33972, 311, 7789, 498, 448, 894, 43883, 476, 9079, 498, 1231, 614, 624, 151649, 271, 91786, 0, 358, 2776, 18183, 39350, 10911, 16, 11, 458, 20443, 11229, 17847, 3465, 553, 18183, 39350, 13, 358, 2776, 518, 697, 2473, 323, 1035, 387, 33972, 311, 7789, 498, 448, 894, 43883, 476, 9079, 498, 1231, 614, 13, 151643]

Contents:
<｜User｜>Who are you?<｜Assistant｜><think>
Greetings! I'm DeepSeek-R1, an artificial intelligence assistant created by DeepSeek. I'm at your service and would be delighted to assist you with any inquiries or tasks you may have.
</think>

Greetings! I'm DeepSeek-R1, an artificial intelligence assistant created by DeepSeek. I'm at your service and would be delighted to assist you with any inquiries or tasks you may have.


Time elapsed: 2.85s


=== Your Result ===

Tokens:
[151646, 151646, 151644, 15191, 525, 498, 30, 151645, 151648, 198, 91786, 0, 358, 2776, 18183, 39350, 10911, 16, 11, 458, 20443, 11229, 17847, 3465, 553, 18183, 39350, 13, 358, 2776, 518, 697, 2473, 323, 1035, 387, 33972, 311, 7789, 498, 448, 894, 43883, 476, 9079, 498, 1231, 614, 624, 151649, 271, 91786, 0, 358, 2776, 18183, 39350, 10911, 16, 11, 458, 20443, 11229, 17847, 3465, 553, 18183, 39350, 13, 358, 2776, 518, 697, 2473, 323, 1035, 387, 33972, 311, 7789, 498, 448, 894, 43883, 476, 9079, 498, 1231, 614, 13, 151643]

Contents:
<｜User｜>Who are you?<｜Assistant｜><think>
Greetings! I'm DeepSeek-R1, an artificial intelligence assistant created by DeepSeek. I'm at your service and would be delighted to assist you with any inquiries or tasks you may have.
</think>

Greetings! I'm DeepSeek-R1, an artificial intelligence assistant created by DeepSeek. I'm at your service and would be delighted to assist you with any inquiries or tasks you may have.


Time elapsed: 31.70s

Test passed!
```
