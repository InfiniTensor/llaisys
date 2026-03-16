target("llaisys-device-cpu")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all")
    if not is_plat("windows") then
        add_cxflags("-Wno-unknown-pragmas")
        -- -fPIC 由根 xmake.lua 在 include 后统一注入，避免本文件内 add_cxflags("-fPIC") 触发 xmake 的 ignore 检查导致构建失败
    end

    add_files("../src/device/cpu/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-ops-cpu")
    set_kind("static")
    add_deps("llaisys-tensor")
    set_languages("cxx17")
    set_warnings("all")
    if not is_plat("windows") then
        add_cxflags("-Wno-unknown-pragmas")
    end

    add_files("../src/ops/*/cpu/*.cpp")

    on_install(function (target) end)
target_end()

