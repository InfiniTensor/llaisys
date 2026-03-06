-- xmake/nvidia.lua
target("llaisys-device-nvidia")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")
    
    -- 启用 CUDA 支持
    add_rules("cuda")
    
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_cuflags("-Xcompiler -fPIC")
    end

    -- 编译 device/nvidia 目录下的源文件
    add_files("../src/device/nvidia/*.cpp", "../src/device/nvidia/*.cu")

    on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia")
    set_kind("static")
    add_deps("llaisys-tensor")
    set_languages("cxx17")
    set_warnings("all", "error")
    
    add_rules("cuda")

    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_cuflags("-Xcompiler -fPIC")
    end

    -- 编译所有 ops 的 nvidia 实现
    add_files("../src/ops/*/nvidia/*.cpp", "../src/ops/*/nvidia/*.cu")

    on_install(function (target) end)
target_end()