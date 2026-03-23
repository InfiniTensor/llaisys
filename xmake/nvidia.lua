target("llaisys-device-nvidia")
    set_kind("static")
    add_deps("llaisys-utils")
    set_languages("cxx17", "cuda")
    set_warnings("all", "error")
    if is_plat("windows") then
        set_runtimes("MD")
        add_cuflags("--compiler-options=/MD", "-rdc=true", {force = true})
    end
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_cuflags("-rdc=true", "--compiler-options=-fPIC")
    end
    add_links("cudart")
    add_links("cudadevrt")
    add_links("nccl")
    add_files("../src/device/nvidia/nvidia_runtime_api.cu")
    add_files("../src/device/nvidia/nvidia_resource.cu")
    add_files("../src/device/nvidia/nvidia_comm.cu")
    on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia")
    set_kind("static")
    add_deps("llaisys-tensor")
    set_languages("cxx17", "cuda")
    set_warnings("all", "error")
    if is_plat("windows") then
        set_runtimes("MD")
        add_cuflags("--compiler-options=/MD", "-rdc=true", {force = true})
    end
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_cuflags("-rdc=true", "--compiler-options=-fPIC")
    end
    add_links("cudart")
    add_links("cudadevrt")
    add_files("../src/ops/*/nvidia/*.cu")
    on_install(function (target) end)
target_end()
