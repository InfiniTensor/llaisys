-- NVIDIA GPU support configuration
-- CUDA files are compiled directly in the main llaisys target to avoid
-- duplicate symbol issues with device linking

target("llaisys-device-nvidia")
    set_kind("static")
    add_deps("llaisys-utils")

    set_toolchains("cuda")
    set_languages("cxx17")
    add_cugencodes("native")
    add_cuflags("-rdc=true", {force = true})

    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_cuflags("-Xcompiler=-fPIC", "-Wno-unknown-pragmas")
    end

    add_includedirs("../include")
    add_includedirs("/usr/include")
    -- CUDA files are compiled in main target, not here
    add_linkdirs("/home/hanson/miniconda3/envs/llaisys/lib/python3.10/site-packages/nvidia/nccl/lib")
    add_syslinks("nccl")

    on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia")
    set_kind("static")
    add_deps("llaisys-tensor")
    add_deps("llaisys-device-nvidia")

    set_toolchains("cuda")
    set_languages("cxx17")
    add_cugencodes("native")
    add_cuflags("-rdc=true", {force = true})

    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_cuflags("-Xcompiler=-fPIC", "-Wno-unknown-pragmas")
    end

    add_includedirs("../include")
    -- CUDA ops files are compiled in main target

    on_install(function (target) end)
target_end()
