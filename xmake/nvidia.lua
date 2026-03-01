-- NVIDIA GPU support configuration

-- Check if NCCL is available (Linux only)
local nccl_available = false
local nccl_lib = nil

if is_plat("linux") then
    -- Try to find NCCL in common locations
    local nccl_paths = {
        "/usr/lib/x86_64-linux-gnu",
        "/usr/local/cuda/lib64",
        os.getenv("NCCL_ROOT") and (os.getenv("NCCL_ROOT") .. "/lib") or nil,
        os.getenv("NCCL_HOME") and (os.getenv("NCCL_HOME") .. "/lib") or nil,
    }
    
    for _, path in ipairs(nccl_paths) do
        if path and os.isfile(path .. "/libnccl.so") then
            nccl_available = true
            nccl_lib = path
            break
        end
    end
end

-- NVIDIA GPU implementation
target("llaisys-device-nvidia")
    set_kind("static")
    -- Device files are added in xmake.lua to avoid duplication
    -- Only add NCCL files if NCCL is available
    if nccl_available then
        add_files("src/device/nvidia/nccl_communicator_impl.cu")
    end
    add_includedirs("$(projectdir)/src")
    
    -- CUDA settings
    add_cugencodes("sm_80")
    add_cuflags("-rdc=true", {force = true})
    
    -- Platform specific
    if is_plat("linux") then
        add_links("cudart", "cublas")
        if nccl_available then
            add_links("nccl")
            add_defines("ENABLE_NCCL")
            if nccl_lib then
                add_linkdirs(nccl_lib)
            end
        end
    else
        add_links("cudart", "cublas")
    end
    
    add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    add_cuflags("-Xcompiler=-fPIC", "-Wno-unknown-pragmas")
    
    on_install(function (target) end)
target_end()

-- NVIDIA operator implementations
-- Note: Operator CUDA files are added in xmake.lua to avoid duplication with llaisys target
target("llaisys-ops-nvidia")
    set_kind("static")
    -- Files are compiled in xmake.lua's llaisys target for proper device linking
    add_includedirs("$(projectdir)/src")
    
    add_cugencodes("sm_80")
    add_cuflags("-rdc=true", {force = true})
    
    if is_plat("linux") then
        add_links("cudart", "cublas")
    else
        add_links("cudart", "cublas")
    end
    
    add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    add_cuflags("-Xcompiler=-fPIC", "-Wno-unknown-pragmas")
    
    on_install(function (target) end)
target_end()

-- Main library with NVIDIA support
target("llaisys-nvidia")
    set_kind("shared")
    add_deps("llaisys-models", "llaisys-ops", "llaisys-ops-cpu", 
             "llaisys-tensor", "llaisys-core", "llaisys-device", 
             "llaisys-device-cpu", "llaisys-utils")
    
    if is_plat("linux") then
        add_deps("llaisys-device-nvidia")
    end
    
    add_linkdirs("/usr/local/cuda/lib64")
    add_links("cudadevrt", "rt", "pthread", "dl")
    
    if is_plat("linux") and nccl_available then
        add_links("nccl")
    end
    
    add_cuflags("-rdc=true", {force = true})
    
    if is_plat("linux") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_cuflags("-Xcompiler=-fPIC", "-Wno-unknown-pragmas")
    end
    
    on_install(function (target) end)
target_end()

-- NCCL availability message
if is_plat("linux") and nccl_available then
    print("NCCL found: " .. (nccl_lib or "system"))
elseif is_plat("linux") then
    print("Warning: NCCL not found. TP will be disabled.")
else
    print("NCCL not available on Windows. TP is disabled.")
end

-- Export NCCL availability for other scripts
if nccl_available then
    set_configvar("NCCL_AVAILABLE", 1)
end

-- Return NCCL status for use in other xmake files
return {
    nccl_available = nccl_available,
    nccl_lib = nccl_lib
}
