-- NVIDIA CUDA 设备层：编译 nvidia_runtime_api.cu 并参与 device-link。
-- 使用前需配置：xmake f --nv-gpu=y [--cuda=/path/to/cuda]
-- xmake 会自动检测 CUDA SDK；也可指定：xmake f --cuda_sdkver=11.8

target("llaisys-device-nvidia")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all")
    add_includedirs("../include", "../src")
    if not is_plat("windows") then
        add_cxflags("-Wno-unknown-pragmas")
        add_cuflags("-Xcompiler=-fPIC")
        add_culdflags("-Xcompiler=-fPIC", {force = true})
    end

    add_files("../src/device/nvidia/*.cu")
    -- 生成与当前主机 SM 兼容的 SASS，以及 PTX 以兼容更多显卡
    add_cugencodes("native")
    add_cugencodes("compute_60")
    -- 静态库含 .cu 时需开启 devlink，否则最终 shared 链接会缺 device 符号
    add_values("cuda.build.devlink", true)

    on_install(function (target) end)
target_end()

-- 算子 CUDA 实现（linear / add / embedding / argmax / rms_norm / rope / swiglu / self_attention）
target("llaisys-ops-nvidia")
    set_kind("static")
    add_deps("llaisys-tensor")
    set_languages("cxx17")
    set_warnings("all")
    add_includedirs("../include", "../src")
    if not is_plat("windows") then
        add_cxflags("-Wno-unknown-pragmas")
        add_cuflags("-Xcompiler=-fPIC")
        add_culdflags("-Xcompiler=-fPIC", {force = true})
    end

    add_files("../src/ops/nvidia/*.cu")
    add_cugencodes("native")
    add_cugencodes("compute_60")
    add_values("cuda.build.devlink", true)

    on_install(function (target) end)
target_end()
