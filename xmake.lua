add_rules("mode.debug", "mode.release")
set_encodings("utf-8")

add_includedirs("include")
add_includedirs("$(builddir)/config")

-- CPU --
includes("xmake/cpu.lua")

-- NVIDIA --
option("nv-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for Nvidia GPU")
option_end()

if has_config("nv-gpu") then
    set_configvar("ENABLE_NVIDIA_API", 1)
    includes("xmake/nvidia.lua")
end

add_configfiles("include/llaisys/build_config.h.in", {prefixdir = "config/llaisys"})

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
    add_options("nv-gpu")
    if has_config("nv-gpu") then
        add_deps("llaisys-device-nvidia")
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
    add_options("nv-gpu")
    if has_config("nv-gpu") then
        add_deps("llaisys-ops-cuda")
    end

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    
    add_files("src/ops/*/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-models")
    set_kind("static")
    add_deps("llaisys-tensor")
    add_deps("llaisys-ops")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/models/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys")
    set_kind("shared")
    add_deps("llaisys-utils")
    add_deps("llaisys-device")
    add_deps("llaisys-core")
    add_deps("llaisys-tensor")
    add_deps("llaisys-ops")
    add_deps("llaisys-models")

    set_languages("cxx17")
    set_warnings("all", "error")
    add_files("src/llaisys/*.cc")
    set_installdir(".")

    if not is_plat("windows") then
        add_ldflags("-fopenmp")
        add_shflags("-fopenmp")
        -- Link OpenBLAS if available (same detection as cpu.lua)
        local candidates = {
            os.getenv("HOME") .. "/.local/lib/python3.10/site-packages/scipy_openblas32",
            os.getenv("HOME") .. "/.local/lib/python3.11/site-packages/scipy_openblas32",
            os.getenv("HOME") .. "/.local/lib/python3.12/site-packages/scipy_openblas32",
        }
        local env_dir = os.getenv("OPENBLAS_DIR")
        if env_dir then
            table.insert(candidates, 1, env_dir)
        end
        for _, base in ipairs(candidates) do
            if os.isdir(base .. "/lib") and os.isfile(base .. "/include/cblas.h") then
                add_linkdirs(base .. "/lib")
                add_rpathdirs(base .. "/lib")
                add_ldflags("-Wl,--no-as-needed -lscipy_openblas -Wl,--as-needed", {force = true})
                add_shflags("-Wl,--no-as-needed -lscipy_openblas -Wl,--as-needed", {force = true})
                break
            end
        end
    end

    
    if has_config("nv-gpu") then
        local cuda_dir = os.getenv("HOME") .. "/.local/cuda"
        if not os.isdir(cuda_dir) then
            cuda_dir = "/usr/local/cuda"
        end
        if os.getenv("CUDA_HOME") then
            cuda_dir = os.getenv("CUDA_HOME")
        end
        add_linkdirs(cuda_dir .. "/lib64")
        add_rpathdirs(cuda_dir .. "/lib64")
        add_ldflags("-Wl,--no-as-needed -lcudart -lcublas -lcublasLt -Wl,--as-needed", {force = true})
        add_shflags("-Wl,--no-as-needed -lcudart -lcublas -lcublasLt -Wl,--as-needed", {force = true})
    end

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