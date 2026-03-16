add_rules("mode.debug", "mode.release")
set_encodings("utf-8")

-- 避免「add_cxflags("-fPIC") is ignored」被当作错误导致构建失败（已对 -fPIC 使用 {force = true}，此策略作兜底）
set_policy("check.auto_ignore_flags", false)

add_includedirs("include")
-- 全局 -fPIC（static 链接进 so 需要），只加一次避免各 target 内 add_cxflags 触发 xmake 的 ignore 检查
if not is_plat("windows") then
    add_cxflags("-fPIC", {force = true})
end

-- CPU --
includes("xmake/cpu.lua")
-- cpu.lua 的 static target 也需 -fPIC，上面全局已加，此处无需再注

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

-- 项目#5：NCCL 张量并行（多 GPU 推理）。需先启用 nv-gpu，并安装 libnccl（如 apt install libnccl2 libnccl-dev 或 conda install nccl）
option("nccl")
    set_default(false)
    set_showmenu(true)
    set_description("Enable NCCL for tensor-parallel distributed inference (Project #5)")
option_end()

if has_config("nccl") and has_config("nv-gpu") then
target("llaisys-nccl")
    set_kind("static")
    add_deps("llaisys-device-nvidia")
    set_languages("cxx17")
    set_warnings("all")
    add_includedirs("include", "src")
    add_defines("ENABLE_NCCL", "ENABLE_NVIDIA_API")
    if not is_plat("windows") then
        add_cuflags("-Xcompiler=-fPIC")
        add_culdflags("-Xcompiler=-fPIC", {force = true})
    end
    add_files("src/llaisys/nccl_comm.cu")
    add_links("nccl")
    add_cugencodes("native")
    add_cugencodes("compute_60")
    add_values("cuda.build.devlink", true)
    on_install(function (target) end)
target_end()
end

target("llaisys-utils")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all")
    if not is_plat("windows") then
        add_cxflags("-Wno-unknown-pragmas")
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
    set_languages("cxx17")
    set_warnings("all")
    if not is_plat("windows") then
        add_cxflags("-Wno-unknown-pragmas")
    end
    add_files("src/device/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-core")
    set_kind("static")
    add_deps("llaisys-utils")
    add_deps("llaisys-device")
    set_languages("cxx17")
    set_warnings("all")
    if not is_plat("windows") then
        add_cxflags("-Wno-unknown-pragmas")
    end
    add_files("src/core/*/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-tensor")
    set_kind("static")
    add_deps("llaisys-core")
    set_languages("cxx17")
    set_warnings("all")
    if not is_plat("windows") then
        add_cxflags("-Wno-unknown-pragmas")
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
    set_languages("cxx17")
    set_warnings("all")
    if not is_plat("windows") then
        add_cxflags("-Wno-unknown-pragmas")
    end
    -- OpenMP：linear 等算子多线程并行，提升 CPU 推理速度（Windows 用 MSVC /openmp，否则 GCC -fopenmp）
    if is_plat("windows") then
        add_cxflags("/openmp")
    else
        add_cxflags("-fopenmp")
        add_mxflags("-fopenmp")
        add_ldflags("-fopenmp")
    end

    -- AVX2+FMА：FP32 linear 使用 SIMD（仅 x86_64）
    if is_arch("x86_64") then
        add_cxflags("-mavx2", "-mfma")
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
    set_warnings("all")
    add_includedirs("src")
    if has_config("nccl") and has_config("nv-gpu") then
        add_deps("llaisys-nccl")
        add_defines("ENABLE_NCCL")
        add_files("src/llaisys/*.cc|nccl_comm_stub.cc")
        add_links("nccl")
    else
        add_files("src/llaisys/*.cc")
    end
    -- OpenMP：Windows 用 MSVC /openmp（无需 gomp.lib），Linux/macOS 用 -fopenmp + gomp
    if is_plat("windows") then
        add_cxflags("/openmp")
        add_ldflags("/openmp")
    else
        add_ldflags("-fopenmp")
        add_links("gomp")
    end
    if has_config("nv-gpu") then
        add_links("cudart")
        add_ldflags("-fPIC")
    end
    set_installdir(".")

    -- 将构建出的动态库复制到 Python 包目录，供 pip install 打包；CI 下 xmake install 后 pip 才能找到 .dll/.so
    after_install(function (target)
        local pkgdir = path.join(os.scriptdir(), "python", "llaisys_py", "libllaisys")
        local built = target:targetfile()
        if os.isfile(built) then
            print("Copying " .. built .. " to python/llaisys_py/libllaisys/ ..")
            os.cp(built, pkgdir)
        end
    end)
target_end()