target("llaisys-device-nvidia")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")

    if not is_plat("windows") then
        add_cxflags("-fPIC")
        add_includedirs("/usr/local/cuda/include")
        add_linkdirs("/usr/local/cuda/lib64")
        add_links("cudart")
    else
        add_includedirs("$(env CUDA_PATH)/include")
        add_linkdirs("$(env CUDA_PATH)/lib/x64")
        add_links("cudart")
    end

    add_files("../src/device/nvidia/*.cpp")

    on_install(function (target) end)
target_end()
