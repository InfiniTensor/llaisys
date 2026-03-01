-- xmake/nvidia.lua

target("llaisys-device-nvidia")
    set_kind("object")
    
    -- 添加 CUDA 编译规则
    add_rules("cuda")
    
    -- ✅ 修改这里：使用 nvcc 的原生参数 -rdc=true
    -- 这会生成可重定位的设备代码，允许链接到共享库
    add_cuflags("-rdc=true", {force = true})
    
    -- 添加源文件
    add_files("$(projectdir)/src/device/nvidia/nvidia_runtime_api.cu")
    -- add_files("$(projectdir)/src/ops/argmax/nvidia/argmax_nvidia.cu")
    
    -- 添加 CUDA 库链接
    add_links("cudart")
    add_defines("ENABLE_NVIDIA_API")
