target("llaisys-device-nvidia")
    set_kind("static")
    set_policy("build.cuda.devlink", true)
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        -- C++ flags for host compiler
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        -- CUDA compiler flags: pass PIC to host compiler and silence pragmas
        add_cuflags("-Xcompiler=-fPIC", "-Wno-unknown-pragmas")
        -- CUDA device-link also needs PIC because this archive is linked into a shared library.
        add_culdflags("-Xcompiler=-fPIC")
    end

    add_files("../src/device/nvidia/*.cu")

    on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia")
    set_kind("static")
    set_policy("build.cuda.devlink", true)
    set_languages("cxx17")
    set_warnings("all", "error")
    add_includedirs("../include")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_cuflags("-Xcompiler=-fPIC", "-Wno-unknown-pragmas")
        add_culdflags("-Xcompiler=-fPIC")
    end

    add_files("../src/ops/*/nvidia/*.cu")

    on_install(function (target) end)
target_end()