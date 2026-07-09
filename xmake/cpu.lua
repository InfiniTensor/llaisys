target("llaisys-device-cpu")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    if has_config("openmp") then
        if is_plat("windows") then
            add_cxflags("/openmp")
        else
            add_cxflags("-fopenmp")
            add_ldflags("-fopenmp")
        end
    end
    if has_config("cpu-avx2") and is_arch("x64", "x86_64") then
        if is_plat("windows") then
            add_cxflags("/arch:AVX2")
        else
            add_cxflags("-mavx2", "-mfma")
        end
    end

    add_files("../src/device/cpu/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-ops-cpu")
    set_kind("static")
    add_deps("llaisys-tensor")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    if has_config("openmp") then
        if is_plat("windows") then
            add_cxflags("/openmp")
        else
            add_cxflags("-fopenmp")
            add_ldflags("-fopenmp")
        end
    end
    if has_config("cpu-avx2") and is_arch("x64", "x86_64") then
        if is_plat("windows") then
            add_cxflags("/arch:AVX2")
        else
            add_cxflags("-mavx2", "-mfma")
        end
    end
    if has_config("openblas") then
        add_defines("ENABLE_OPENBLAS")
        add_packages("openblas")
    end

    add_files("../src/ops/*/cpu/*.cpp")

    on_install(function (target) end)
target_end()

