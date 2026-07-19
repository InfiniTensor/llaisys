add_rules("mode.debug", "mode.release")
set_encodings("utf-8")

add_includedirs("include")

-- 全局开启 OpenMP 支持、最高级别优化
-- native 模式 (服务器): 使用 -march=native, 编译器自动启用 AVX-512 等本机指令集
-- 默认模式 (本地):       显式指定 -mavx2 -mfma, 兼容大多数 x86-64 CPU
add_cxflags("-fopenmp", "-O3")
add_ldflags("-fopenmp")
add_shflags("-fopenmp")
add_syslinks("gomp") -- 显式链接 GNU OpenMP 库

option("native")
    set_default(false)
    set_showmenu(true)
    set_description("Use -march=native for best performance on current CPU (enables AVX-512 on supported CPUs)")
option_end()

if has_config("native") then
    add_cxflags("-march=native")
else
    add_cxflags("-mavx2", "-mfma")
end

-- OpenBLAS 集成: 从源码编译安装到 ~/openblas
option("openblas")
    set_default(true)
    set_showmenu(true)
    set_description("Whether to use OpenBLAS for linear algebra acceleration")
option_end()

if has_config("openblas") then
    add_defines("USE_OPENBLAS")
    add_includedirs(os.getenv("HOME") .. "/openblas/include")
    add_linkdirs(os.getenv("HOME") .. "/openblas/lib")
    add_links("openblas")
    add_rpathdirs(os.getenv("HOME") .. "/openblas/lib")
end

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

    set_languages("cxx17")
    set_warnings("all", "error")
    add_files("src/llaisys/*.cc")
    add_files("src/models/*.cpp")
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