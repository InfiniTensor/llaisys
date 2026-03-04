target("llaisys-device-nvidia")
    set_kind("static")
    add_deps("llaisys-utils")

    set_policy("build.cuda.devlink", true)
    set_toolchains("cuda")
    add_links("cudart")
    add_cugencodes("native")

    on_load(function (target)
        import("lib.detect.find_tool")
        local nvcc = find_tool("nvcc")
        if nvcc ~= nil then
            if is_plat("windows") then
                nvcc_path = os.iorun("where nvcc"):match("(.-)\r?\n")
            else
                nvcc_path = nvcc.program
            end

            target:add("linkdirs", path.directory(path.directory(nvcc_path)) .. "/lib64/stubs")
            target:add("links", "cuda")
        end
    end)

    if is_plat("windows") then
        add_cuflags("-Xcompiler=/utf-8", "--expt-relaxed-constexpr", "--allow-unsupported-compiler")
        add_cuflags("-Xcompiler=/W3", "-Xcompiler=/WX")
        add_cxxflags("/FS")
    else
        add_cuflags("-Xcompiler=-Wall", "-Xcompiler=-Werror")
        add_cuflags("-Xcompiler=-fPIC")
        add_cuflags("--extended-lambda")
        add_culdflags("-Xcompiler=-fPIC")
        add_cxxflags("-fPIC")
        add_cuflags("--expt-relaxed-constexpr")
    end

    add_cuflags("-Xcompiler=-Wno-error=deprecated-declarations")
    add_cuflags("-Xcompiler=-Wno-unknown-pragmas")

    set_languages("cxx17")
    set_warnings("all", "error")
    
    add_files("../src/**/nvidia/*.cu")

    on_install(function (target) end)
target_end()