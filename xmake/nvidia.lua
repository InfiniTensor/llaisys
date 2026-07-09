target("llaisys-device-nvidia")
    set_kind("static")
    set_languages("cxx17", "cuda")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_cuflags("-Xcompiler=-fPIC")
    end
    add_cuflags("-rdc=true")

    add_files("../src/device/nvidia/*.cu")
    add_files("../src/device/nvidia/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia")
    set_kind("static")
    add_deps("llaisys-tensor")
    set_languages("cxx17", "cuda")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_cuflags("-Xcompiler=-fPIC")
    end
    add_cuflags("-rdc=true")

    add_files("../src/ops/*/nvidia/*.cu")
    add_files("../src/ops/*/nvidia/*.cpp")

    on_install(function (target) end)
target_end()
