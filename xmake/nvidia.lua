target("llaisys-device-nvidia")
    set_kind("static")
    add_deps("llaisys-utils")

    -- 强制开启 CUDA 设备代码链接策略
    set_policy("build.cuda.devlink", true)
    
    set_toolchains("cuda")
    add_links("cudart", "cublas")
    add_cugencodes("native")

    -- 动态查找 CUDA 路径并链接基础库
    on_load(function (target)
        import("lib.detect.find_tool")
        local nvcc = find_tool("nvcc")
        if nvcc ~= nil then
            local nvcc_path = nvcc.program
            target:add("linkdirs", path.directory(path.directory(nvcc_path)) .. "/lib64/stubs")
            target:add("links", "cuda")
        end
    end)

    if not is_plat("windows") then
        add_cuflags("-Xcompiler=-Wall", "-Xcompiler=-Werror")
        add_cuflags("-Xcompiler=-fPIC")
        add_cuflags("--extended-lambda")
        add_culdflags("-Xcompiler=-fPIC")
        add_cxxflags("-fPIC")
    end

    set_languages("cxx17")
    set_warnings("all", "error")
    
    add_files("../src/device/nvidia/*.cu")
    add_files("../src/ops/*/nvidia/*.cu")

    on_install(function (target) end)
target_end()