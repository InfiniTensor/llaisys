# Metax 第二平台设计稿

## 1. 目标
- 满足课程项目 2 “支持第二种平台” 的设计说明要求。
- 本次仓库不提交 Metax 可执行代码，只给出后端扩展方案。

## 2. 设备抽象扩展
- 在 `llaisysDeviceType_t` 中增加 `LLAISYS_DEVICE_METAX`。
- 在 `src/device/runtime_api.hpp` 中新增 `metax::getRuntimeAPI()`。
- 在 `xmake.lua` 中增加 `--metax-gpu=y` 开关。

## 3. Runtime 映射
- `get_device_count`
- `set_device`
- `device_synchronize`
- `create_stream` / `destroy_stream`
- `malloc_device` / `free_device`
- `malloc_host` / `free_host`
- `memcpy_sync` / `memcpy_async`

以上接口与当前 CPU / NVIDIA runtime 完全对齐，这样上层 `tensor / ops / model` 不需要重写。

## 4. 算子目录组织
- `src/ops/add/metax/`
- `src/ops/argmax/metax/`
- `src/ops/embedding/metax/`
- `src/ops/linear/metax/`
- `src/ops/rms_norm/metax/`
- `src/ops/rope/metax/`
- `src/ops/self_attention/metax/`
- `src/ops/swiglu/metax/`

调度方式与 NVIDIA 一样，通过各算子的 `op.cpp` 做设备分发。

## 5. 实现顺序
1. 先打通 runtime API 和最小 tensor 分配/拷贝。
2. 再实现 `add` / `argmax` / `embedding` 这类易验证算子。
3. 再实现 `linear` 与 `rms_norm`。
4. 最后补 `rope`、`self_attention`、`swiglu` 和模型推理链路。

## 6. 验收策略
- 先通过 `test/test_runtime.py --device metax`
- 再通过 `test/test_ops.py --device metax`
- 最后通过 `test/test_infer.py --device metax --test`

## 7. 风险点
- Metax 数学库能力是否支持大矩阵乘和半精度
- stream / event 语义是否与 CUDA 一致
- host/device memcpy 是否支持异步
- Python 侧推理速度是否需要额外 graph / compile 优化
