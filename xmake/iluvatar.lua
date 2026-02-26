-- Iluvatar (天数智芯) GPU support configuration
-- This file defines separate static libraries for Iluvatar device and ops
-- which are then linked into the main llaisys shared library

target("llaisys-device-iluvatar")
    set_kind("static")
    add_deps("llaisys-utils")

    set_toolchains("cuda")
    set_languages("cxx17")
    add_cugencodes("native")
    add_cuflags("-rdc=true", {force = true})

    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_cuflags("-Xcompiler=-fPIC", {force = true})
    end

    add_includedirs("../include")
    add_files("../src/device/iluvatar/*.cu")

    on_install(function (target) end)
target_end()

target("llaisys-ops-iluvatar")
    set_kind("static")
    add_deps("llaisys-tensor")
    add_deps("llaisys-device-iluvatar")

    set_toolchains("cuda")
    set_languages("cxx17")
    add_cugencodes("native")
    add_cuflags("-rdc=true", {force = true})

    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_cuflags("-Xcompiler=-fPIC", {force = true})
    end

    add_includedirs("../include")
    add_files("../src/ops/*/iluvatar/*.cu")

    on_install(function (target) end)
target_end()
