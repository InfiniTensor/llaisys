target("llaisys-device-cpu")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")
    if is_plat("windows") then
        add_cxflags("/wd4267")  -- 禁用 'size_t to int' 警告
    else
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("../src/device/cpu/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-ops-cpu")
    set_kind("static")
    add_deps("llaisys-tensor")
    set_languages("cxx17")
    set_warnings("all", "error")
    if is_plat("windows") then
        add_cxflags("/wd4267")  -- 禁用 'size_t to int' 警告
    else
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("../src/ops/*/cpu/*.cpp")

    on_install(function (target) end)
target_end()
