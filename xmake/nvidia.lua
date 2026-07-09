target("llaisys-device-nvidia")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")
    local cuda_sdk = get_config("cuda")
    if cuda_sdk then
        local cuda_include = path.join(cuda_sdk, "include")
        local cuda_lib64 = path.join(cuda_sdk, "lib64")
        local cuda_lib = path.join(cuda_sdk, "lib")
        if os.isdir(cuda_include) then
            add_includedirs(cuda_include)
        end
        if os.isdir(cuda_lib64) then
            add_linkdirs(cuda_lib64)
        end
        if os.isdir(cuda_lib) then
            add_linkdirs(cuda_lib)
        end
    end
    add_links("cudart")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("../src/device/nvidia/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia")
    set_kind("static")
    add_deps("llaisys-tensor")

    add_rules("cuda")
    set_values("cuda.rdc", false)

    set_languages("cxx17")
    set_warnings("all", "error")
    local cuda_sdk = get_config("cuda")
    if cuda_sdk then
        local cuda_include = path.join(cuda_sdk, "include")
        local cuda_lib64 = path.join(cuda_sdk, "lib64")
        local cuda_lib = path.join(cuda_sdk, "lib")
        if os.isdir(cuda_include) then
            add_includedirs(cuda_include)
        end
        if os.isdir(cuda_lib64) then
            add_linkdirs(cuda_lib64)
        end
        if os.isdir(cuda_lib) then
            add_linkdirs(cuda_lib)
        end
    end
    add_links("cudart", "cublas")
    if not is_plat("windows") then
        add_cuflags("-Xcompiler=-fPIC", {force = true})
    end

    add_files("../src/ops/*/nvidia/*.cu")

    on_install(function (target) end)
target_end()
