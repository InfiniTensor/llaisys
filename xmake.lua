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

option("sentencepiece")
    set_default(false)
    set_showmenu(true)
    set_description("Enable SentencePiece tokenizer support")
option_end()

if has_config("nv-gpu") then
    add_defines("ENABLE_NVIDIA_API")
    includes("xmake/nvidia.lua")
end

-- ILUVATAR --
option("iluvatar-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for Iluvatar CoreX GPU")
option_end()

if has_config("iluvatar-gpu") then
    add_defines("ENABLE_ILUVATAR_API")
    includes("xmake/iluvatar.lua")
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
    if has_config("iluvatar-gpu") then
        add_deps("llaisys-device-iluvatar")
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
    if has_config("iluvatar-gpu") then
        add_deps("llaisys-ops-iluvatar")
    end

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
    add_files("src/llaisys/*/*.cpp")
    add_files("src/models/*/*.cpp")
    add_files("src/models/*/*/*.cpp")
    add_files("src/tokenizer/*/*.cpp")
    set_installdir(".")

    if has_config("sentencepiece") then
        add_defines("LLAISYS_ENABLE_SENTENCEPIECE")
        add_links("sentencepiece")
    end
    if has_config("nv-gpu") then
        set_languages("cxx17", "cuda")
        set_policy("build.cuda.devlink", true)
        add_links("cudadevrt", "cudart")
        add_files("src/device/nvidia/devlink_stub.cu")
    elseif has_config("iluvatar-gpu") then
        -- No .cu files in this target, no CUDA toolchain
        -- Use add_shflags to control exact link order:
        -- 1. whole-archive iluvatar static libs (defines nvidia:: symbols)
        -- 2. -lcudart AFTER the .a files (so cudart symbols are resolved)
        add_shflags(
            "-Wl,--whole-archive",
            "build/linux/x86_64/release/libllaisys-ops-iluvatar.a",
            "build/linux/x86_64/release/libllaisys-device-iluvatar.a",
            "-Wl,--no-whole-archive",
            "-L/usr/local/corex/lib64",
            "-Wl,-rpath,/usr/local/corex/lib64",
            "-lcudart",
            {force = true}
        )
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