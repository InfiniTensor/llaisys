add_rules("mode.debug", "mode.release")
set_encodings("utf-8")

add_includedirs("include")

-- CPU --
includes("xmake/cpu.lua")

-- NVIDIA --
option("openblas")
    set_default(false)
    set_showmenu(true)
    set_description("Use OpenBLAS for linear (matmul) on CPU; install libopenblas-dev and run xmake f --openblas=y")
option_end()

option("nv-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for Nvidia GPU")
option_end()

option("mx-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for MetaX GPU")
option_end()

if has_config("nv-gpu") then
    add_defines("ENABLE_NVIDIA_API")
    includes("xmake/nvidia.lua")
end

if has_config("mx-gpu") then
    add_defines("ENABLE_METAX_API")
    includes("xmake/metax.lua")
end

target("llaisys-utils")
    set_kind("static")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/utils/*.cpp")

    on_install(function (target) end)
target_end()


target("llaisys-device")
    set_kind("static")
    add_deps("llaisys-utils")
    add_deps("llaisys-device-cpu")
    if has_config("nv-gpu") then
        add_deps("llaisys-device-nvidia")
    end
    if has_config("mx-gpu") then
        add_deps("llaisys-device-metax")
    end

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
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

    add_files("src/tensor/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-ops")
    set_kind("static")
    add_deps("llaisys-ops-cpu")
    if has_config("nv-gpu") then
        add_deps("llaisys-ops-nvidia")
    end
    if has_config("mx-gpu") then
        add_deps("llaisys-ops-metax")
        -- Propagate metax operator archive to final link step in dependency order.
        add_links("llaisys-ops-metax")
    end

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    
    add_files("src/ops/*/*.cpp")

    on_install(function (target) end)
target_end()

if has_config("nv-gpu") then
    target("llaisys-ops-nvidia")
        set_kind("static")
        add_deps("llaisys-tensor")

        set_languages("cxx17")
        set_warnings("all", "error")
        add_files("src/ops/*/nvidia/*.cu")
        add_includedirs("include", "src")

        -- CUDA arch targets (keep simple; adjust later for perf/compat)
        add_cugencodes("native")
        add_cugencodes("compute_75")

        -- Ensure static lib does CUDA devlink once (because final .so has no .cu)
        add_values("cuda.build.devlink", true)

        if not is_plat("windows") then
            add_cxflags("-fPIC", "-Wno-unknown-pragmas")
            -- nvcc compile + devlink must be PIC
            add_cuflags("-Xcompiler -fPIC", "-Xcompiler -Wno-unknown-pragmas")
            add_culdflags("-Xcompiler -fPIC", "-Xcompiler -Wno-unknown-pragmas")
        end

        on_install(function (target) end)
    target_end()
end

target("llaisys")
    set_kind("shared")
    add_deps("llaisys-utils")
    add_deps("llaisys-device")
    add_deps("llaisys-core")
    add_deps("llaisys-tensor")
    add_deps("llaisys-ops")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_ldflags("-fopenmp")
        add_syslinks("gomp")
    end
    if has_config("nv-gpu") then
        add_syslinks("cudart")
        add_syslinks("cublas")
    end
    add_files("src/llaisys/*.cc")
    add_files("src/models/qwen2/*.cpp")
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
