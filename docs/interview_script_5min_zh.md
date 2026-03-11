# 5 分钟讲稿（项目 2 第二平台 MetaX/MACA）

大家好，我这次汇报的是 LLAISYS 项目 2 第二平台的落地工作。原来的分支已经完成了 CPU 和 NVIDIA 路径，但第二平台当时还只是 Metax 设计稿。我的目标，是把它推进成沐曦平台上真正可编译、可运行、可测试的实现。

我一开始没有直接写代码，而是先做平台识别。仓库根目录有一份沐曦平台说明 PDF，但真正决定实现方式的还是本机 SDK。我在机器上检查了 `mx-smi`、`mxcc --version`、`/opt/maca/include`、`/opt/maca/lib` 和 `/opt/mxdriver/lib`，最后确认这台机器是 `MetaX C500`，`MACA 3.2.1.10`，驱动是 `3.0.11`，编译器是 `mxcc 1.0.0`。

接下来最关键的判断是兼容性。结论不是一句“兼容 CUDA”就能说完。C++ SDK 层面，MetaX 不是 CUDA drop-in，因为它用的是 `mc_runtime` 和 `mcblas`，不是 `cuda_runtime` 和 `cuBLAS`，编译器也不是 `nvcc`。但在 Python 和 PyTorch 这一层，它又保留了 CUDA 语义兼容，我实测 `torch.cuda.is_available()` 是真的，设备名也能读到 `MetaX C500`。所以最终方案不是复用 NVIDIA 后端，而是新增独立 `METAX` 后端，同时让 Hugging Face 对照继续走 `torch.cuda`。

在实现上，我先扩展了设备抽象，新增 `LLAISYS_DEVICE_METAX`，然后在构建系统里加了 `--metax-gpu=y` 和独立的 `xmake/metax.lua`，统一用 `mxcc` 编译。runtime 这一层，我把设备数量、设备切换、stream、malloc/free、memcpy 这些通用接口，完整映射到了 `mcGetDeviceCount`、`mcSetDevice`、`mcStreamCreateWithFlags`、`mcMalloc`、`mcMemcpy` 这一套 MACA API。

算子部分我采取的是最小可落地策略。`add`、`embedding`、`rms_norm`、`rope`、`swiglu` 我做成了 MetaX kernel；`linear` 直接接 `mcblasGemmEx`，这是推理里最关键的算子；`argmax` 和 `self_attention` 先保留 host fallback，优先把整条链路打通。这样做的原因很简单，这次提交的重点是让第二平台达到“可测试”，而不是一次性把所有算子都做成高性能版本。

验证顺序我也是按主链路来的。先构建，再跑 `test_runtime.py --device metax`，然后跑 `test_ops.py --device metax`，最后跑 `test_infer.py --device metax --test`。因为本地没有现成可提交的模型目录，我用了一个公开的小模型 `trl-internal-testing/tiny-Qwen2ForCausalLM-2.5` 做严格一致性测试。最后 CPU 基线、MetaX 的 runtime、ops、infer 都通过了，说明这条链路已经真正可用。

这里还有两个比较典型的工程点。第一，`xmake` 在 root 下会直接拒绝运行，所以构建命令必须显式加 `XMAKE_ROOT=y`。第二，我没有把 MetaX 逻辑塞进 NVIDIA 分支，而是全程做独立目录和独立开关，这样可以最大限度降低对 CPU 和 NVIDIA 现有路径的回归风险。

如果用一句话总结，我这次完成的工作不是再补一份第二平台说明文档，而是把 MetaX/MACA 真正接进了 LLAISYS，并在真实沐曦机器上把 `runtime -> ops -> infer` 主链路跑到了可提交状态。
