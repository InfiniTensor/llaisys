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

    add_files("src/utils/*.cpp")

    on_install(function (target) end)
target_end()


target("llaisys-device")
    set_kind("static")
    add_deps("llaisys-utils")
    add_deps("llaisys-device-cpu")
    -- Note: CUDA device files are compiled directly into the llaisys shared lib
    -- to avoid __cudaRegisterLinkedBinary_* symbols being dropped by the linker.

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
    -- Note: CUDA ops files are compiled directly into the llaisys shared lib.

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
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

    -- 直接将 .cu 文件编译进共享库，避免中间静态库导致
    -- __cudaRegisterLinkedBinary_* 符号被链接器丢弃的问题（cuda.devlink 老版本不支持）。
    if has_config("nv-gpu") then
        add_rules("cuda")
        add_cuflags("--generate-code=arch=compute_80,code=sm_80", {force = true})
        add_cuflags("-std=c++17")
        if not is_plat("windows") then
            add_cuflags("-Xcompiler=-fPIC,-Wno-unknown-pragmas")
        end
        add_files("src/device/nvidia/*.cu")
        add_files("src/ops/*/nvidia/*.cu")
        add_links("cublas", "cuda", "cudart")
    end

    set_languages("cxx17")
    set_warnings("all", "error")
    add_files("src/llaisys/*.cc")
    set_installdir(".")

    
    after_install(function (target)
        -- copy shared library to python package source tree
        print("Copying llaisys to python/llaisys/libllaisys/ ..")
        if is_plat("windows") then
            os.cp("bin/*.dll", "python/llaisys/libllaisys/")
        end
        if is_plat("linux") then
            os.cp("lib/*.so", "python/llaisys/libllaisys/")
        end

        -- (re-)install the Python package so that site-packages picks up the new .so
        -- Using --no-build-isolation avoids re-running cmake/setup.py build steps.
        print("Installing Python package (pip install -e python/) ..")
        local ret = os.execv("pip", {"install", "-e", "python/", "--no-build-isolation", "-q"})
        if ret ~= 0 then
            -- Fallback: try pip3
            os.execv("pip3", {"install", "-e", "python/", "--no-build-isolation", "-q"})
        end
        print("Python package installed. You can now run the tests.")
    end)
target_end()