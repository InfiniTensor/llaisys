add_rules("mode.debug", "mode.release")
set_encodings("utf-8")

add_includedirs("include")

-- CPU --
includes("xmake/cpu.lua")

-- NVIDIA --
option("nv-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for Nvidia GPU")
option_end()

option("openmp")
    set_default(true)
    set_showmenu(true)
    set_description("Whether to enable OpenMP for CPU operators")
option_end()

option("cpu-avx2")
    set_default(true)
    set_showmenu(true)
    set_description("Whether to enable AVX2/FMA for CPU operators")
option_end()

option("openblas")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to enable OpenBLAS backend for CPU linear f32")
option_end()

if has_config("openblas") then
    add_requires("openblas", {optional = true})
end

if has_config("nv-gpu") then
    add_defines("ENABLE_NVIDIA_API")
    includes("xmake/nvidia.lua")
end

target("llaisys-utils")
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

    add_files("src/utils/*.cpp")

    on_install(function (target) end)
target_end()


target("llaisys-device")
    set_kind("static")
    add_deps("llaisys-utils")
    add_deps("llaisys-device-cpu")

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

    add_files("src/device/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-core")
    set_kind("static")
    add_deps("llaisys-utils")
    add_deps("llaisys-device")

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

    add_files("src/core/*/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-tensor")
    set_kind("static")
    add_deps("llaisys-core")

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

    add_files("src/tensor/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-ops")
    set_kind("static")
    add_deps("llaisys-ops-cpu")

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
    
    add_files("src/ops/*/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys")
    set_kind("shared")
    add_deps("llaisys-utils")
    add_deps("llaisys-device")
    add_deps("llaisys-core")
    add_deps("llaisys-tensor")
    add_deps("llaisys-ops")

    set_languages("cxx17")
    set_warnings("all", "error")
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
    add_files("src/llaisys/*.cc")
    set_installdir(".")

    
    after_install(function (target)
        -- copy shared library to python package
        print("Copying llaisys to python/llaisys/libllaisys/ ..")
        if is_plat("windows") then
            os.cp("bin/*.dll", "python/llaisys/libllaisys/")
        end
        if is_plat("linux") then
            os.cp("lib/*.so", "python/llaisys/libllaisys/")
        end
    end)
target_end()