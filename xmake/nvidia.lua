-- NVIDIA GPU 设备：CUDA Runtime API + 资源
-- 使用方式: xmake f --nv-gpu=y [--cuda=/path/to/cuda]
target("llaisys-device-nvidia")
    set_kind("static")
    add_deps("llaisys-utils")
    set_languages("cxx17")
    add_files("../src/device/nvidia/*.cu")
    add_cugencodes("native")
    add_cugencodes("compute_75")
    add_values("cuda.build.devlink", true)
    add_includedirs("../include", "../src")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        -- nvcc: pass -fPIC to host compiler and to devlink step (for _gpucode.cu.o)
        add_cuflags("-Xcompiler -fPIC", "-Xcompiler -Wno-unknown-pragmas")
        add_culdflags("-Xcompiler -fPIC", "-Xcompiler -Wno-unknown-pragmas")
    end
    on_install(function (target) end)
target_end()
