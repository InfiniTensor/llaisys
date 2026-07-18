add_rules("mode.debug", "mode.release")
set_encodings("utf-8")

add_includedirs("include")

local function has_gpu_backend()
    return has_config("nv-gpu") or has_config("mx-gpu")
end

-- CPU --
includes("xmake/cpu.lua")

-- NVIDIA --
option("nv-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for Nvidia GPU")
option_end()

-- MetaX / MACA --
option("mx-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for MetaX GPU with MACA")
option_end()

option("maca-path")
    set_default("/opt/maca")
    set_showmenu(true)
    set_description("MACA toolkit path")
option_end()

option("mxdriver-path")
    set_default("/opt/mxdriver")
    set_showmenu(true)
    set_description("MetaX driver path")
option_end()

local use_nv_gpu = has_config("nv-gpu")
local use_mx_gpu = has_config("mx-gpu")

if use_nv_gpu and use_mx_gpu then
    raise("Please enable only one GPU backend at a time: nv-gpu or mx-gpu")
end

if use_nv_gpu then
    add_defines("ENABLE_NVIDIA_API")
    includes("xmake/nvidia.lua")

    if is_plat("linux") then
        add_sysincludedirs("/usr/local/cuda/include")
        add_linkdirs("/usr/local/cuda/lib64")
    end
elseif use_mx_gpu then
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
        add_links("llaisys-device-metax")
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
    add_deps("llaisys-device-cpu")
    add_deps("llaisys-ops-cpu")

    set_languages("cxx17")
    set_warnings("all", "error")

    if has_config("nv-gpu") then
        add_deps("llaisys-device-nvidia")
        add_deps("llaisys-ops-nvidia")
    end
    if has_config("mx-gpu") then
        add_deps("llaisys-device-metax")
        add_deps("llaisys-ops-metax")
        add_links("llaisys-device-metax", "llaisys-ops-metax")
    end

    if has_config("nv-gpu") then
        -- Use nvcc as the shared library linker for proper CUDA device code linking
        set_toolset("sh", "nvcc")

        if is_plat("linux") then
            add_syslinks("cudart", "cublas")
            add_shflags("-Xcompiler", "-fPIC", "-shared", "-rdc=true", {force = true})
        end
    end

    if has_config("mx-gpu") and is_plat("linux") then
        local maca_path = get_config("maca-path") or "/opt/maca"
        local mxdriver_path = get_config("mxdriver-path") or "/opt/mxdriver"
        add_linkdirs(path.join(maca_path, "lib"))
        add_linkdirs(path.join(mxdriver_path, "lib"))
        add_syslinks("mcruntime", "mxc-runtime64", "runtime_cu", "mxsml", "mcblas")
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
        if is_plat("macosx") then
            os.cp("lib/*.dylib", "python/llaisys/libllaisys/")
        end
    end)
target_end()
