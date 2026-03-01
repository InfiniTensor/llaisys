-- xmake/nvidia.lua

target("llaisys-device-nvidia")
    -- 设置类型为对象文件集合 (object)，这样它可以被链接到其他目标中
    set_kind("object")
    
    -- 添加 CUDA 编译规则
    -- 这会自动配置 nvcc 编译器、CUDA 头文件路径等
    add_rules("cuda")
    
    -- 添加源文件
    -- 这里包含我们之前写的 Runtime API 和 Argmax 算子的 .cu 文件
    -- 如果有更多 .cu 文件，继续往这里加
    add_files("$(projectdir)/src/device/nvidia/nvidia_runtime_api.cu")
    -- 注意：如果你还没准备好 argmax，可以先注释掉下面这行，避免警告
    -- add_files("$(projectdir)/src/ops/argmax/nvidia/argmax_nvidia.cu")
    
    -- 添加 CUDA 库链接
    -- cudart 是 CUDA Runtime 的核心库
    add_links("cudart")
    
    -- 添加宏定义
    -- 这个宏会开启代码中的 #ifdef ENABLE_NVIDIA_API 分支
    -- (注意：根目录已经定义了，这里其实可以省略，但加上也没坏处)
    add_defines("ENABLE_NVIDIA_API")
