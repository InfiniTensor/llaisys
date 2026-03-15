target("llaisys-device-cpu")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("../src/device/cpu/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-ops-cpu")
    set_kind("static")
    add_deps("llaisys-tensor")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_cxflags("-fopenmp")
    elseif is_plat("windows") then
        add_cxflags("/openmp")
    end
    if has_config("openblas") then
        add_defines("LLAISYS_USE_OPENBLAS")
        add_links("openblas")
        add_syslinks("openblas")
        -- 常见 cblas 头路径（按需取消注释或添加本机路径）
        add_includedirs("/usr/include/x86_64-linux-gnu", "/usr/include", {public = false})
    end

    add_files("../src/ops/*/cpu/*.cpp")

    on_install(function (target) end)
target_end()

