target("llaisys-device-nvidia")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")
    set_policy("build.cuda.devlink", true)

    add_files("../src/device/nvidia/*.cu")

    add_cugencodes("compute_80")
    add_cuflags("--use_fast_math")
    add_cuflags("--objdir-as-tempdir")
    add_cuflags("-Xcompiler=-fPIC")
    add_culdflags("-Xcompiler=-fPIC")

    add_links("cudart")
    add_linkdirs("/usr/local/cuda/lib64")

    on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia")
    set_kind("static")
    add_deps("llaisys-tensor", "llaisys-device-nvidia")
    set_languages("cxx17")
    set_warnings("all", "error")
    set_policy("build.cuda.devlink", true)

    add_files("../src/ops/*/cuda/*.cu")

    add_cugencodes("compute_80")
    add_cuflags("--use_fast_math")
    add_cuflags("--objdir-as-tempdir")
    add_cuflags("-Xcompiler=-fPIC")
    add_culdflags("-Xcompiler=-fPIC")

    add_links("cudart")
    add_linkdirs("/usr/local/cuda/lib64")

    on_install(function (target) end)
target_end()
