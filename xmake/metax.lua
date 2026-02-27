-- MetaX GPU support configuration
-- This file defines separate static libraries for MetaX device and ops
-- which are then linked into the main llaisys shared library

target("llaisys-device-metax")
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
    add_files("../src/device/metax/*.cu")

    on_install(function (target) end)
target_end()

target("llaisys-ops-metax")
    set_kind("static")
    add_deps("llaisys-tensor")
    add_deps("llaisys-device-metax")

    set_toolchains("cuda")
    set_languages("cxx17")
    add_cugencodes("native")
    add_cuflags("-rdc=true", {force = true})

    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_cuflags("-Xcompiler=-fPIC", "-Wno-unknown-pragmas")
    end

    add_includedirs("../include")
    add_files("../src/ops/*/metax/*.cu")

    on_install(function (target) end)
target_end()
