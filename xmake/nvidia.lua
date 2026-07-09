local cuda_dir = os.getenv("HOME") .. "/.local/cuda"
if not os.isdir(cuda_dir) then
    cuda_dir = "/usr/local/cuda"
end
if os.getenv("CUDA_HOME") then
    cuda_dir = os.getenv("CUDA_HOME")
end

local cuda_include = cuda_dir .. "/include"
local cuda_lib = cuda_dir .. "/lib64"
local nvcc = cuda_dir .. "/bin/nvcc"

local cuda_flags = {
    "-std=c++17", "--expt-relaxed-constexpr", "-O3",
    "--compiler-options=-fPIC,-Wno-unknown-pragmas",
    "-m64", "-gencode", "arch=compute_86,code=sm_86",
    "-DNDEBUG",
    "-Iinclude", "-Ibuild/config", "-I" .. cuda_include,
}

rule("cu_nordc")
    set_extensions(".cu")
    on_buildcmd_file(function (target, batchcmds, sourcefile, opt)
        local objectfile = target:objectfile(sourcefile)
        batchcmds:mkdir(path.directory(objectfile))
        local args = table.join(cuda_flags, {"-c", "-o", objectfile, sourcefile})
        batchcmds:show("compiling.cuda %s", sourcefile)
        batchcmds:vrunv(nvcc, args)
        batchcmds:add_depfiles(sourcefile)
        table.insert(target:objectfiles(), objectfile)
    end)
rule_end()

target("llaisys-device-nvidia")
    set_kind("static")
    add_rules("cu_nordc")

    add_files("../src/device/nvidia/*.cu")

    on_install(function (target) end)
target_end()

target("llaisys-ops-cuda")
    set_kind("static")
    add_deps("llaisys-tensor")
    add_rules("cu_nordc")

    add_files("../src/ops/*/cuda/*.cu")

    on_install(function (target) end)
target_end()
