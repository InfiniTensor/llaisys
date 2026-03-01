-- xmake/nvidia.lua

target("llaisys-device-nvidia")
    set_kind("object")
    
    -- 添加 CUDA 编译规则
    add_rules("cuda")
    
    -- ✅ 修改为 add_cuflags (这是 Xmake 中给 .cu 文件传参的标准方式)
    add_cuflags("-fPIC")
    
    -- 添加源文件
    add_files("$(projectdir)/src/device/nvidia/nvidia_runtime_api.cu")
    -- 如果还没准备好 argmax，可以先注释掉
    -- add_files("$(projectdir)/src/ops/argmax/nvidia/argmax_nvidia.cu")
    
    -- 添加 CUDA 库链接
    add_links("cudart")
    
    -- 添加宏定义
    add_defines("ENABLE_NVIDIA_API")
