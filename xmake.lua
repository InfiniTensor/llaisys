add_rules("mode.debug", "mode.release")
set_encodings("utf-8")

add_requires("openmp")
add_requires("openblas") -- 自动处理 Windows/Linux 差异
add_packages("openmp", "openblas")
add_includedirs("include")

-- CPU / GPU 配置保持不变...
includes("xmake/cpu.lua")

-- ==========================================
-- 1. 静态组件：使用你的 on_install 技巧防止硬塞系统目录
-- ==========================================
local static_targets = {"llaisys-utils", "llaisys-device", "llaisys-core", "llaisys-tensor", "llaisys-ops"}

for _, name in ipairs(static_targets) do
    target(name)
        set_kind("static")
        set_languages("cxx17")
        if not is_plat("windows") then add_cxflags("-fPIC") end
        
        -- 核心：直接套用你的方案
        on_install(function (target) end) 

        -- 根据名字添加特定的文件和依赖
        if name == "llaisys-utils" then add_files("src/utils/*.cpp") end
        if name == "llaisys-device" then 
            add_deps("llaisys-utils", "llaisys-device-cpu")
            add_files("src/device/*.cpp") 
        end
        if name == "llaisys-core" then 
            add_deps("llaisys-utils", "llaisys-device")
            add_files("src/core/*/*.cpp") 
        end
        if name == "llaisys-tensor" then 
            add_deps("llaisys-core")
            add_files("src/tensor/*.cpp") 
        end
        if name == "llaisys-ops" then 
            add_deps("llaisys-ops-cpu")
            add_packages("openmp", "openblas")
            add_files("src/ops/*/*.cpp") 
        end
    target_end()
end

-- ==========================================
-- 2. 主动态库：只安装到本地
-- ==========================================
target("llaisys")
    set_kind("shared")
    add_deps("llaisys-utils", "llaisys-device", "llaisys-core", "llaisys-tensor", "llaisys-ops")
    add_packages("openmp", "openblas")

    add_files("src/llaisys/*.cpp", "src/llaisys/*.cc")
    set_languages("cxx17")
    set_installdir(".") -- 强制安装到当前目录

    after_install(function (target)
        print("Copying llaisys to python/llaisys/libllaisys/ ..")
        local lib_dir = "python/llaisys/libllaisys/"
        if is_plat("windows") then
            os.cp("bin/*.dll", lib_dir)
        else
            os.cp("lib/*.so", lib_dir)
        end
    end)
target_end()