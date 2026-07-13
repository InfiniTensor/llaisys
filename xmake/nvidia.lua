local function config_nvidia_target()
    add_rules("cuda")
    set_values("cuda.build.devlink", true)
    set_languages("cxx17")
    set_warnings("all", "error")
    add_cugencodes("native")
    add_cuflags("--extended-lambda", {force = true})
    add_cuflags("--expt-relaxed-constexpr", {force = true})
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_cuflags("-Xcompiler=-fPIC", {force = true})
        add_culdflags("-Xcompiler -fPIC", {force = true})
    end
    add_links("cudart", "cublas", "nccl")
end

target("llaisys-device-nvidia")
    set_kind("static")
    config_nvidia_target()
    add_files("../src/device/nvidia/*.cu")

    on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia")
    set_kind("static")
    add_deps("llaisys-tensor")
    config_nvidia_target()

    for _, file in ipairs(os.files("../src/ops/*/nvidia/*.cu")) do
        add_files(file)
    end

    on_install(function (target) end)
target_end()
