# LLAISYS CUDA 适配报告

## 1. 项目目标与范围

本报告记录了 LLAISYS 在 NVIDIA CUDA 平台上的适配工作，覆盖内容包括：

- 前置依赖安装
- CUDA 构建与安装流程
- CUDA Runtime API 与算子实现
- Qwen2 GPU 推理接入与 CPU 兼容性说明
- 复现与验证方法

本次适配遵循原则：

1. 不改变现有 CPU 推理接口语义。
2. CUDA 路径以算子原生 kernel 为主，不采用“CPU 计算后拷回 GPU”的默认策略。
3. Python 层 API 与现有调用方式保持兼容。

---

## 2. 前置依赖安装

### 2.1 系统与工具链

建议环境（本次验证环境）：

- Linux x86_64
- NVIDIA GPU（A100，SM80）
- CUDA Toolkit 12.x（本仓库验证为 12.8）
- GCC / G++（支持 C++17）
- Xmake
- Python 3.9+

### 2.2 Python 依赖

在仓库根目录执行：

```bash
python3 -m pip install -e ./python/llaisyscore --user --break-system-packages
python3 -m pip install -e ./python/server-project --user --break-system-packages
```

如需推理测试（含 HF 对照），还需确保环境中可用：

- torch
- transformers
- accelerate
- safetensors
- huggingface-hub

---

## 3. 编译与安装

### 3.1 CUDA 开关编译

```bash
xmake f --nv-gpu=y -cv
xmake -v
xmake install
```

其中 `xmake install` 会将 `libllaisys.so` 同步到 `python/llaisys/libllaisys/`。

### 3.2 构建系统关键配置

文件：`xmake.lua`

```lua
option("nv-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for Nvidia GPU")
option_end()

if has_config("nv-gpu") then
    add_defines("ENABLE_NVIDIA_API")
    includes("xmake/nvidia.lua")
end
```

文件：`xmake/nvidia.lua`

```lua
target("llaisys-device-nvidia")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")
    set_policy("build.cuda.devlink", true)

    add_files("../src/device/nvidia/*.cu")

    add_cugencodes("compute_80")
    add_cuflags("--use_fast_math")
    add_cuflags("-Xcompiler=-fPIC")
    add_culdflags("-Xcompiler=-fPIC")

    add_links("cudart")
    add_linkdirs("/usr/local/cuda/lib64")

target_end()
```

说明：

- `build.cuda.devlink` + `-Xcompiler=-fPIC` 解决了 CUDA device link 与共享库链接问题。
- `compute_80` 与当前测试硬件匹配。

---

## 4. CUDA Runtime API 实现

文件：`src/device/nvidia/nvidia_runtime_api.cu`

主要实现了：

- `getDeviceCount` / `setDevice`
- stream 创建、销毁、同步
- `cudaMalloc/cudaFree`、`cudaMallocHost/cudaFreeHost`
- `memcpySync/memcpyAsync` 的 H2H/H2D/D2H/D2D 映射

代码片段（仓库实现）：

```cpp
int getDeviceCount() {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    return count;
}

void *mallocDevice(size_t size) {
    void *ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));
    return ptr;
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    cudaMemcpyKind cuda_kind = cudaMemcpyDefault;
    switch (kind) {
    case LLAISYS_MEMCPY_H2H:
        cuda_kind = cudaMemcpyHostToHost;
        break;
    case LLAISYS_MEMCPY_H2D:
        cuda_kind = cudaMemcpyHostToDevice;
        break;
    case LLAISYS_MEMCPY_D2H:
        cuda_kind = cudaMemcpyDeviceToHost;
        break;
    case LLAISYS_MEMCPY_D2D:
        cuda_kind = cudaMemcpyDeviceToDevice;
        break;
    default:
        ASSERT(false, "Unsupported memcpy kind in CUDA memcpySync");
    }
    CUDA_CHECK(cudaMemcpy(dst, src, size, cuda_kind));
}
```

---

## 5. CUDA 算子实现

### 5.1 通用 dtype 读写工具

文件：`src/utils/cuda_type_utils.cuh`

该工具统一支持 F32/F16/BF16 的 device 端读取与写回，避免各算子重复实现类型分发。

```cpp
__device__ __forceinline__ float load_as_float(const void *ptr, size_t idx, llaisysDataType_t dtype) {
    if (dtype == LLAISYS_DTYPE_F32) return load_as_float_f32(ptr, idx);
    if (dtype == LLAISYS_DTYPE_F16) return load_as_float_f16(ptr, idx);
    return load_as_float_bf16(ptr, idx);
}
```

### 5.2 add（F32/F16/BF16）

文件：`src/ops/add/cuda/add_cuda.cu`

```cpp
template <typename T>
__global__ void add_kernel(T *c, const T *a, const T *b, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    c[idx] = a[idx] + b[idx];
}
```

对于 FP16/BF16 使用对应半精度运算指令。

### 5.3 embedding

文件：`src/ops/embedding/cuda/embedding_cuda.cu`

```cpp
__global__ void embedding_kernel(void *out, const int64_t *index, const void *weight,
                                 llaisysDataType_t dtype, size_t num, size_t dim) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = num * dim;
    if (idx >= total) return;

    size_t row = idx / dim;
    size_t col = idx % dim;
    int64_t token = index[row];
    size_t src_idx = static_cast<size_t>(token) * dim + col;
    float val = load_as_float(weight, src_idx, dtype);
    store_from_float(out, idx, val, dtype);
}
```

同时在 `src/ops/embedding/op.cpp` 中补充了 GPU index 越界校验时的 D2H 安全路径（避免直接把 device 指针当 host 指针解引用）。

### 5.4 linear

文件：`src/ops/linear/cuda/linear_cuda.cu`

```cpp
__global__ void linear_kernel(void *out, const void *in, const void *weight, const void *bias,
                              llaisysDataType_t type, size_t batch_size, size_t in_features,
                              size_t out_features, bool has_bias) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch_size * out_features;
    if (idx >= total) return;

    size_t b = idx / out_features;
    size_t o = idx % out_features;

    float sum = 0.0f;
    for (size_t i = 0; i < in_features; ++i) {
        float x = load_as_float(in, b * in_features + i, type);
        float w = load_as_float(weight, o * in_features + i, type);
        sum += x * w;
    }
    if (has_bias) sum += load_as_float(bias, o, type);
    store_from_float(out, idx, sum, type);
}
```

### 5.5 argmax

文件：`src/ops/argmax/cuda/argmax_cuda.cu`

```cpp
__global__ void argmax_kernel(int64_t *mi, void *mv, const void *v, llaisysDataType_t type, size_t numel) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    size_t best_idx = 0;
    float best_val = load_as_float(v, 0, type);
    for (size_t i = 1; i < numel; ++i) {
        float cur = load_as_float(v, i, type);
        if (cur > best_val) {
            best_val = cur;
            best_idx = i;
        }
    }
    mi[0] = static_cast<int64_t>(best_idx);
    store_from_float(mv, 0, best_val, type);
}
```

### 5.6 rms_norm

文件：`src/ops/rms_norm/cuda/rms_norm_cuda.cu`

```cpp
__global__ void rms_norm_kernel(void *out, const void *in, const void *weight,
                                llaisysDataType_t type, size_t feature_dim, float eps) {
    size_t row = blockIdx.x;
    extern __shared__ float sdata[];

    float local_sum = 0.0f;
    for (size_t col = threadIdx.x; col < feature_dim; col += blockDim.x) {
        float val = load_as_float(in, row * feature_dim + col, type);
        local_sum += val * val;
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        __syncthreads();
    }

    float inv_rms = rsqrtf(sdata[0] / static_cast<float>(feature_dim) + eps);
    for (size_t col = threadIdx.x; col < feature_dim; col += blockDim.x) {
        float x = load_as_float(in, row * feature_dim + col, type);
        float w = load_as_float(weight, col, type);
        store_from_float(out, row * feature_dim + col, x * w * inv_rms, type);
    }
}
```

### 5.7 rope

文件：`src/ops/rope/cuda/rope_cuda.cu`

```cpp
__global__ void rope_kernel(void *out, const void *in, const int64_t *pos_ids, llaisysDataType_t type,
                            size_t seq_len, size_t n_heads, size_t head_dim, float theta) {
    // ... 计算 phi/cos/sin 后旋转写回
}
```

### 5.8 self_attention

文件：`src/ops/self_attention/cuda/self_attention_cuda.cu`

```cpp
__global__ void self_attention_kernel(void *attn_val, const void *q, const void *k, const void *v,
                                      llaisysDataType_t type, size_t seqlen, size_t totlen,
                                      size_t nh, size_t nkvh, size_t d, size_t dv, float scale) {
    // 单 kernel 内完成 causal masked attention 的分数、归一化与加权求和
}
```

### 5.9 swiglu

文件：`src/ops/swiglu/cuda/swiglu_cuda.cu`

```cpp
__global__ void swiglu_kernel(void *out, const void *gate, const void *up,
                              llaisysDataType_t type, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    float g = load_as_float(gate, idx, type);
    float u = load_as_float(up, idx, type);
    float sig = g / (1.0f + expf(-g));
    store_from_float(out, idx, u * sig, type);
}
```

### 5.10 random_sample（temperature/top-k/top-p）

文件：`src/ops/random_sample/cuda/random_sample_cuda.cu`

实现流程：

1. logits 温度缩放
2. device 侧排序（thrust）
3. 按 top-k/top-p 截断
4. 使用 `curand` 采样

```cpp
curandStatePhilox4_32_10_t rng_state;
curand_init(seed, 0ULL, 0ULL, &rng_state);
float r = curand_uniform(&rng_state);
```

并通过 host 侧原子计数 + 时间戳组合 seed，避免固定采样序列。

---

## 6. Qwen2 CUDA 推理支持

### 6.1 Python 侧模型设备接入

文件：`python/llaisys/models/qwen2.py`

- 支持 `DeviceType.NVIDIA` 初始化
- CUDA 路径权重 dtype 使用 BF16
- 权重加载时保持 host tensor，使用 `tensorLoad` 进行 H2D

关键代码：

```python
def maybe_cast_tensor(tensor):
    if self.device == DeviceType.CPU:
        return tensor.to(torch.float32).contiguous()
    elif self.device == DeviceType.NVIDIA:
        return tensor.to(torch.bfloat16).contiguous()
    return tensor
```

此处避免 `.cuda()`，防止把 device 指针走到 `tensorLoad(H2D)` 的 host 输入路径造成不兼容。

### 6.2 端到端验证结果

已通过：

- `python test/test_infer.py --device nvidia --max_steps 128`
- `python test/test_infer.py --device nvidia --max_steps 8 --test`

其中 `--test` 模式已与 HF 在确定性配置下对齐并 `Test passed`。

---

## 7. 复现方法（可直接执行）

### 7.1 编译与安装

```bash
xmake f --nv-gpu=y -cv
xmake -v
xmake install
```

### 7.2 运行时测试

```bash
python test/test_runtime.py --device nvidia
```

### 7.3 算子测试（NVIDIA）

```bash
for f in test/ops/*.py; do
  echo "===== RUN $f --device nvidia ====="
  python "$f" --device nvidia
  echo
 done
```

### 7.4 Tensor 双设备迁移补充测试

```bash
python test/test_tensor.py --device nvidia
python test/test_tensor_dual_device.py --device_id 0
```

### 7.5 Qwen2 推理测试

```bash
python test/test_infer.py --device nvidia --max_steps 128
python test/test_infer.py --device nvidia --max_steps 8 --test
```

---

## 8. 兼容性说明

1. CPU 路径保持可用，CUDA 仅在 `--nv-gpu=y` + `ENABLE_NVIDIA_API` 条件下启用。
2. Python API 兼容既有调用方式，不要求业务侧改接口。
3. 采样接口未新增外部 seed 参数；CUDA 侧 seed 为实现细节，不影响现有测试接口。

---

## 9. 已知限制与后续优化方向

1. 当前部分 CUDA kernel 仍是基础实现（可进一步做块内优化、向量化、融合）。
2. 与 HF 相比 decode 吞吐仍有差距，后续可优先优化：
   - 减少每 token 动态分配
   - 优化采样（减少全排序开销）
   - 注意力 kernel 融合/flash 化
   - 进一步减少同步与跨设备拷贝

---

## 10. 本次交付结论

本次已经完成从 Runtime 到算子到模型推理的 CUDA 全链路接入：

- 构建可编译、可安装
- `test_runtime` 与 `test/ops` 在 NVIDIA 设备通过
- `Qwen2` 在 NVIDIA 设备可完成推理，并通过 `test_infer --test` 的一致性验证

即：LLAISYS 已具备可复现的 CUDA 推理能力，且与 CPU 实现保持接口兼容。
