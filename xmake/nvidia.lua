-- xmake/nvidia.lua

-- 定义目标：llaisys-device-nvidia
-- 名字必须与根目录 xmake.lua 中的 add_deps("llaisys-device-nvidia") 完全一致
target("llaisys-device-nvidia")
    -- 设置类型为对象文件集合 (object)
    set_kind("object")
    
    -- 添加 CUDA 编译规则
    add_rules("cuda")
    
    -- ✅ 关键修改：使用 nvcc 原生参数 -rdc=true 生成可重定位设备代码
    -- 这解决了链接到共享库 (.so) 时的 relocation 错误
    -- {force = true} 强制 Xmake 应用此标志，防止被忽略
    add_cuflags("-rdc=true", {force = true})
    
    -- 添加源文件
    -- 这里包含我们之前写的 Runtime API
    add_files("$(projectdir)/src/device/nvidia/nvidia_runtime_api.cu")
    
    -- 如果有其他 .cu 算子文件，可以在这里继续添加
    -- 例如: add_files("$(projectdir)/src/ops/argmax/nvidia/argmax_nvidia.cu")
    -- 注意：如果文件不存在，Xmake 会报警告，可以暂时注释掉
    
    -- 添加 CUDA 库链接
    add_links("cudart")
    
    -- 添加宏定义
    add_defines("ENABLE_NVIDIA_API")
