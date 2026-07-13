add_rules("mode.debug", "mode.release")
set_encodings("utf-8")

add_includedirs("include")

-- CPU --
includes("xmake/cpu.lua")

-- NVIDIA --
option("nv-gpu")
    set_default(true)
    set_showmenu(true)
    set_description("Whether to compile implementations for Nvidia GPU")
option_end()

if has_config("nv-gpu") then
    add_defines("ENABLE_NVIDIA_API")
    includes("xmake/nvidia.lua")

    -- 强制注入 fPIC 兜底
    local nvidia_target = target("llaisys-device-nvidia")
    if nvidia_target then
        nvidia_target:add("cxflags", "-fPIC", {force = true})
        nvidia_target:add("cuflags", "-Xcompiler=-fPIC", {force = true})
        nvidia_target:add("culdflags", "-Xcompiler=-fPIC", {force = true})
    end
end

target("llaisys-utils")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all")
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
    set_warnings("all")
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
    set_warnings("all")
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
    set_warnings("all")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    add_files("src/tensor/*.cpp")
    on_install(function (target) end)
target_end()

target("llaisys-ops")
    set_kind("static")
    add_deps("llaisys-ops-cpu")
    
    -- 【修复点】：彻底移除了对 llaisys-ops-nvidia 的依赖，防止报错

    set_languages("cxx17")
    set_warnings("all")
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
    set_warnings("all")
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

    if has_config("nv-gpu") then
        add_rules("cuda")
        if not is_plat("windows") then
            add_cuflags("-Xcompiler=-fPIC")
        end
        -- 【核心逻辑】：直接把所有算子的 cuda 文件喂给这个拥有一切依赖的动态库
        add_files("src/ops/*/nvidia/*.cu")
    end
    
    set_languages("cxx17")
    set_warnings("all")
    add_files("src/llaisys/*.cc")

    set_installdir(".")

    after_install(function (target)
        print("Copying llaisys to python/llaisys/libllaisys/ ..")
        if is_plat("windows") then
            os.cp("bin/*.dll", "python/llaisys/libllaisys/")
        end
        if is_plat("linux") then
            os.cp("lib/*.so", "python/llaisys/libllaisys/")
        end
    end)
target_end()