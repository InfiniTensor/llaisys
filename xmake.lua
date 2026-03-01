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

-- MetaX --
option("metax-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for MetaX GPU")
option_end()

if has_config("metax-gpu") then
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

    add_files("src/models/*/*.cpp")
    add_includedirs("/usr/include")  -- For NCCL headers

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

    if has_config("nv-gpu") then
        -- Add CUDA files directly to main target for proper device linking
        add_files("src/device/nvidia/*.cu")
        add_files("src/ops/*/nvidia/*.cu")
        add_linkdirs("/usr/local/cuda/lib64")
        add_linkdirs("/home/hanson/miniconda3/envs/llaisys/lib/python3.10/site-packages/nvidia/nccl/lib")
        add_syslinks("cudart", "cublas", "nccl")
        add_shflags("-Wl,--no-as-needed", "-lcudart", "-lcublas", "-lnccl", {force = true})
        add_shflags("-Wl,-rpath,/home/hanson/miniconda3/envs/llaisys/lib/python3.10/site-packages/nvidia/nccl/lib", {force = true})
        set_toolchains("cuda")
        add_cugencodes("native")
        add_cuflags("-rdc=true", {force = true})
        add_includedirs("/usr/include")  -- For NCCL headers
    end

    if has_config("metax-gpu") then
        -- Directly add CUDA object files instead of static libraries to avoid RDC issues
        add_files("src/device/metax/*.cu")
        add_files("src/ops/*/metax/*.cu")
        add_linkdirs("/opt/maca/lib", "/opt/maca/tools/cu-bridge/lib")
        add_syslinks("mcblas", "mcruntime", "cuda")
        add_shflags("-Wl,--no-as-needed", "-lmcblas", "-lmcruntime", "-lcuda", {force = true})
        set_toolchains("cuda")
        add_cugencodes("native")
        -- Disable device link for MetaX
        set_policy("cuda.devlink", false)
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
