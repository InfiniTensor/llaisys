option("cpu-blas")
    set_default(true)
    set_showmenu(true)
    set_description("Whether to enable OpenBLAS acceleration for large CPU linear kernels when available")
option_end()

option("openblas-prefix")
    set_default("")
    set_showmenu(true)
    set_description("Prefix directory of OpenBLAS installation (expects include/cblas.h and lib/libopenblas.so)")
option_end()

local _openblas_config = false
local _openblas_checked = false
local _openblas_warned = false

local function _detect_openblas_from_prefix(prefix)
    if not prefix or #prefix == 0 then
        return nil
    end

    local include_dirs = {
        path.join(prefix, "include"),
        path.join(prefix, "include", "x86_64-linux-gnu"),
    }
    local lib_dirs = {
        path.join(prefix, "lib"),
        path.join(prefix, "lib64"),
        path.join(prefix, "lib", "x86_64-linux-gnu"),
    }

    for _, include_dir in ipairs(include_dirs) do
        if not os.isfile(path.join(include_dir, "cblas.h")) then
            goto continue
        end
        for _, lib_dir in ipairs(lib_dirs) do
            if os.isfile(path.join(lib_dir, "libopenblas.so")) then
                return {
                    include_dir = include_dir,
                    lib_dir = lib_dir,
                }
            end
        end
        ::continue::
    end

    return nil
end

local function _get_openblas_config()
    if _openblas_checked then
        return _openblas_config
    end
    _openblas_checked = true

    if not has_config("cpu-blas") then
        _openblas_config = nil
        return _openblas_config
    end

    local prefixes = {}
    local configured_prefix = get_config("openblas-prefix")
    if configured_prefix and #configured_prefix > 0 then
        table.insert(prefixes, configured_prefix)
    end

    local conda_prefix = os.getenv("CONDA_PREFIX")
    if conda_prefix and #conda_prefix > 0 then
        table.insert(prefixes, conda_prefix)
    end

    table.insert(prefixes, "/usr")
    table.insert(prefixes, "/usr/local")

    local seen = {}
    for _, prefix in ipairs(prefixes) do
        if seen[prefix] then
            goto continue
        end
        seen[prefix] = true

        local config = _detect_openblas_from_prefix(prefix)
        if config ~= nil then
            _openblas_config = config
            return _openblas_config
        end
        ::continue::
    end

    _openblas_config = nil
    return _openblas_config
end

function add_cpu_blas_settings()
    if not has_config("cpu-blas") then
        return
    end

    local config = _get_openblas_config()
    if config == nil then
        if not _openblas_warned then
            print("warning: OpenBLAS not found, falling back to internal CPU linear kernels")
            _openblas_warned = true
        end
        return
    end

    add_defines("LLAISYS_USE_OPENBLAS")
    add_includedirs(config.include_dir)
    add_linkdirs(config.lib_dir)
    add_links("openblas")
    if not is_plat("windows") then
        add_rpathdirs(config.lib_dir)
    end
end

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
        add_cxflags("-fPIC", "-Wno-unknown-pragmas", "-fopenmp")
    else
        add_cxflags("/openmp")
    end
    add_cpu_blas_settings()

    add_files("../src/ops/*/cpu/*.cpp")

    on_install(function (target) end)
target_end()
