add_rules("mode.debug", "mode.release")
set_encodings("utf-8")

add_includedirs("include")

-- CPU --
includes("xmake/cpu.lua")

-- NVIDIA --
option("nv-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for Nvidia GPU")
option_end()

-- MetaX/MACA --
option("metax-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for MetaX/MACA GPU")
option_end()

if has_config("nv-gpu") and has_config("metax-gpu") then
    raise("nv-gpu and metax-gpu cannot be enabled together in the same build")
end

if has_config("nv-gpu") then
    add_defines("ENABLE_NVIDIA_API")
    includes("xmake/nvidia.lua")
end

if has_config("metax-gpu") then
    add_defines("ENABLE_METAX_API")
    includes("xmake/metax.lua")
end

target("llaisys-utils")
    set_kind("static")

    set_languages("cxx17")
    set_warnings("all", "error")
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
    if has_config("metax-gpu") then
        add_deps("llaisys-device-metax")
    end

    set_languages("cxx17")
    set_warnings("all", "error")
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
    set_warnings("all", "error")
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
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/tensor/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-ops")
    set_kind("static")
    add_deps("llaisys-ops-cpu")
    if has_config("nv-gpu") then
        add_deps("llaisys-ops-nvidia")
    end
    if has_config("metax-gpu") then
        add_deps("llaisys-ops-metax")
    end

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    
    add_files("src/ops/*/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys")
    set_kind("shared")
    add_deps("llaisys-utils")
    add_deps("llaisys-device")
    add_deps("llaisys-core")
    add_deps("llaisys-tensor")
    add_deps("llaisys-ops")
    if has_config("nv-gpu") then
        add_rules("cuda")
        add_files("src/llaisys/cuda_link_stub.cu")
        add_links("cudart", "cublas", "cudadevrt")
    end
    if has_config("metax-gpu") then
        add_rules("metax")
        add_values("metax.files", "src/llaisys/metax_link_stub.cu")
        add_linkdirs(os.getenv("MACA_LIBDIR") or "/opt/maca/lib")
        add_linkdirs(os.getenv("MXDRIVER_LIBDIR") or "/opt/mxdriver/lib")
        add_links("mcruntime", "mcblas")
        add_rpathdirs(os.getenv("MACA_LIBDIR") or "/opt/maca/lib")
        add_rpathdirs(os.getenv("MXDRIVER_LIBDIR") or "/opt/mxdriver/lib")
        on_link(function (target, opt)
            local mxcc = get_config("mxcc") or os.getenv("MXCC") or "/opt/maca/mxgpu_llvm/bin/mxcc"
            local argv = {"-shared", "-o", target:targetfile()}

            for _, objectfile in ipairs(target:objectfiles()) do
                table.insert(argv, objectfile)
            end
            for _, dep in ipairs(target:orderdeps()) do
                if dep:kind() == "static" then
                    table.insert(argv, dep:targetfile())
                end
            end
            for _, dir in ipairs(target:get("linkdirs") or {}) do
                table.insert(argv, "-L" .. dir)
            end
            for _, dir in ipairs(target:get("rpathdirs") or {}) do
                table.insert(argv, "-Wl,-rpath," .. dir)
            end
            for _, link in ipairs(target:get("links") or {}) do
                table.insert(argv, "-l" .. link)
            end
            for _, link in ipairs(target:get("syslinks") or {}) do
                table.insert(argv, "-l" .. link)
            end
            for _, flag in ipairs(target:get("shflags") or {}) do
                table.insert(argv, flag)
            end

            os.mkdir(path.directory(target:targetfile()))
            os.vrunv(mxcc, argv)
        end)
    end

    add_files("src/llaisys/models/qwen2.cpp")
    if os.isfile("src/llaisys/models/llama.cpp") then
        add_files("src/llaisys/models/llama.cpp")
    end

    set_languages("cxx17")
    set_warnings("all", "error")
    add_files("src/llaisys/*.cc")
    set_installdir(".")
    if not is_plat("windows") then
        add_ldflags("-fopenmp", "-lgomp")
        add_shflags("-fopenmp", "-lgomp")
    end

    
    after_install(function (target)
        -- copy shared library to python package
        print("Copying llaisys to python/llaisys/libllaisys/ ..")
        if is_plat("windows") then
            os.cp("bin/*.dll", "python/llaisys/libllaisys/")
        end
        if is_plat("linux") then
            os.cp("lib/*.so", "python/llaisys/libllaisys/")
        end
    end)
target_end()
