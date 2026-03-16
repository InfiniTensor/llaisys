target("llaisys-device-cpu")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("../src/device/cpu/*.cpp")

    on_install(function (target) end)
target_end()

-- Detect OpenBLAS from scipy_openblas32 Python package
local use_openblas = false
local openblas_include_dir = nil
local openblas_lib_dir = nil

if not is_plat("windows") then
    -- Try known paths for scipy_openblas32
    local candidates = {
        os.getenv("HOME") .. "/.local/lib/python3.10/site-packages/scipy_openblas32",
        os.getenv("HOME") .. "/.local/lib/python3.11/site-packages/scipy_openblas32",
        os.getenv("HOME") .. "/.local/lib/python3.12/site-packages/scipy_openblas32",
        "/usr/lib/python3/dist-packages/scipy_openblas32",
    }

    -- Also check OPENBLAS_DIR env
    local env_dir = os.getenv("OPENBLAS_DIR")
    if env_dir then
        table.insert(candidates, 1, env_dir)
    end

    for _, base in ipairs(candidates) do
        if os.isfile(base .. "/include/cblas.h") and os.isdir(base .. "/lib") then
            openblas_include_dir = base .. "/include"
            openblas_lib_dir = base .. "/lib"
            use_openblas = true
            print("OpenBLAS detected: " .. openblas_lib_dir)
            break
        end
    end

    if not use_openblas then
        -- Check system paths
        if os.isfile("/usr/include/cblas.h") or os.isfile("/usr/include/x86_64-linux-gnu/cblas.h") then
            use_openblas = true
            openblas_include_dir = "/usr/include"
            openblas_lib_dir = "/usr/lib/x86_64-linux-gnu"
            print("System OpenBLAS detected")
        else
            print("OpenBLAS not found, using built-in optimized GEMM")
        end
    end
end

target("llaisys-ops-cpu")
    set_kind("static")
    add_deps("llaisys-tensor")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas", "-fopenmp", "-mavx2", "-mfma", "-O3")
        if use_openblas then
            add_defines("USE_OPENBLAS")
            add_includedirs(openblas_include_dir)
        end
    else
        add_cxflags("/openmp", "/arch:AVX2", "/O2")
    end

    add_files("../src/ops/*/cpu/*.cpp")

    on_install(function (target) end)
target_end()
