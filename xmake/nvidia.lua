-- xmake/nvidia.lua

-- 定义目标：llaisys-device-nvidia
-- 名字必须与根目录 xmake.lua 中的 add_deps("llaisys-device-nvidia") 一致
target("llaisys-device-nvidia")
    -- 设置类型为对象文件集合 (object)
    set_kind("object")
    
    -- 添加 CUDA 编译规则
    add_rules("cuda")
    
    -- ✅ 关键修改：强制添加 -fPIC 标志，以便能被链接到共享库 (.so) 中
    add_cuflags("-fPIC", {force = true})
    
    -- 添加源文件
    -- 这里包含我们之前写的 Runtime API
    add_files("$(projectdir)/src/device/nvidia/nvidia_runtime_api.cu")
    
    -- ⚠️ 注意：如果你还没完成 argmax 算子，请保持下面这行注释掉，避免编译警告或错误
    -- add_files("$(projectdir)/src/ops/argmax/nvidia/argmax_nvidia.cu")
    
    -- 添加 CUDA 库链接
    add_links("cudart")
    
    -- 添加宏定义
    add_defines("ENABLE_NVIDIA_API")
