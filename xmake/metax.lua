-- MetaX GPU backend integration.
-- Usage: xmake f --mx-gpu=y

local function _append_unique(list, value)
    if not value or value == "" then
        return
    end
    for _, item in ipairs(list) do
        if item == value then
            return
        end
    end
    table.insert(list, value)
end

local function _metax_roots()
    local roots = {}
    _append_unique(roots, os.getenv("MACA_HOME"))
    _append_unique(roots, "/opt/maca")
    _append_unique(roots, "/usr/local/maca")
    _append_unique(roots, "/opt/maca-3.3.0")
    _append_unique(roots, "/opt/maca-3.2.0")
    _append_unique(roots, "/opt/maca-3.1.0")
    return roots
end

local function _metax_include_dirs()
    local dirs = {}
    for _, root in ipairs(_metax_roots()) do
        local d1 = path.join(root, "include")
        local d2 = path.join(root, "include", "mcr")
        local d3 = path.join(root, "mxgpu_llvm", "include")
        if os.isdir(d1) then _append_unique(dirs, d1) end
        if os.isdir(d2) then _append_unique(dirs, d2) end
        if os.isdir(d3) then _append_unique(dirs, d3) end
    end
    return dirs
end

local function _metax_link_dirs()
    local dirs = {}
    for _, root in ipairs(_metax_roots()) do
        local d1 = path.join(root, "lib")
        local d2 = path.join(root, "lib64")
        local d3 = path.join(root, "mxgpu_llvm", "lib")
        local d4 = path.join(root, "mxgpu_llvm", "lib64")
        if os.isdir(d1) then _append_unique(dirs, d1) end
        if os.isdir(d2) then _append_unique(dirs, d2) end
        if os.isdir(d3) then _append_unique(dirs, d3) end
        if os.isdir(d4) then _append_unique(dirs, d4) end
    end
    return dirs
end

local function _apply_metax_search_paths(target)
    for _, includedir in ipairs(_metax_include_dirs()) do
        target:add("includedirs", includedir, {public = true})
    end
    for _, linkdir in ipairs(_metax_link_dirs()) do
        target:add("linkdirs", linkdir, {public = true})
    end
end

local function _resolve_mxcc()
    local mxcc = os.getenv("MXCC")
    if mxcc and mxcc ~= "" then
        return mxcc
    end
    local maca_home = os.getenv("MACA_HOME")
    if maca_home and maca_home ~= "" then
        local candidate = path.join(maca_home, "mxgpu_llvm", "bin", "mxcc")
        if os.isfile(candidate) then
            return candidate
        end
    end
    return "mxcc"
end

target("llaisys-device-metax")
    set_kind("static")
    add_deps("llaisys-utils")
    set_languages("cxx17")
    set_warnings("all", "error")

    -- Keep .maca as canonical source files, but compile wrappers for xmake 2.8.x compatibility.
    on_load(function (target)
        local projectdir = os.projectdir()
        local gen_dir = path.join(projectdir, "build", "_gen", "metax")
        os.mkdir(gen_dir)

        local maca_sources = {
            path.join(projectdir, "src", "device", "metax", "metax_resource.maca"),
            path.join(projectdir, "src", "device", "metax", "metax_runtime_api.maca")
        }

        for _, source in ipairs(maca_sources) do
            local base = path.basename(source)
            local wrap = path.join(gen_dir, base .. "_wrapper.cpp")
            io.writefile(wrap, "#include \"" .. path.translate(source) .. "\"\n")
            target:add("files", wrap)
        end

        _apply_metax_search_paths(target)
    end)

    add_includedirs("../include", "../src")

    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    -- Link common runtime library names shipped by MACA.
    add_syslinks("mcruntime", "mxc-runtime64", "runtime_cu", {public = true})

    on_install(function (target) end)
target_end()

target("llaisys-ops-metax")
    set_kind("static")
    add_deps("llaisys-tensor")
    set_languages("cxx17")
    set_warnings("all", "error")

    on_load(function (target)
        local projectdir = os.projectdir()
        local obj_dir = path.join(projectdir, "build", "_gen", "metax_ops_obj")
        os.mkdir(obj_dir)

        local maca_sources = os.files(path.join(projectdir, "src", "ops", "*", "metax", "*.maca"))
        local objectfiles = {}
        for _, source in ipairs(maca_sources) do
            local op_name = path.basename(path.directory(path.directory(source)))
            local base = path.basename(source)
            local objectfile = path.join(obj_dir, op_name .. "_" .. base .. ".o")
            table.insert(objectfiles, objectfile)
        end

        target:data_set("metax_maca_sources", maca_sources)
        target:data_set("metax_maca_objectfiles", objectfiles)
        _apply_metax_search_paths(target)
    end)

    -- Build .maca sources via mxcc manually to avoid xmake 2.8.x toolscript limitations.
    on_build(function (target)
        local projectdir = os.projectdir()
        local mxcc = _resolve_mxcc()
        local include_dirs = {
            path.join(projectdir, "include"),
            path.join(projectdir, "src")
        }
        for _, includedir in ipairs(_metax_include_dirs()) do
            table.insert(include_dirs, includedir)
        end

        local sources = target:data("metax_maca_sources") or {}
        local objectfiles = target:data("metax_maca_objectfiles") or {}
        for i, source in ipairs(sources) do
            local objectfile = objectfiles[i]
            os.mkdir(path.directory(objectfile))

            local args = {
                "-std=c++17",
                "-O3",
                "-fPIC",
                "-Wno-unknown-pragmas",
                "-DENABLE_METAX_API"
            }
            for _, includedir in ipairs(include_dirs) do
                table.insert(args, "-I" .. includedir)
            end
            table.insert(args, "-c")
            table.insert(args, source)
            table.insert(args, "-o")
            table.insert(args, objectfile)

            os.vrunv(mxcc, args)
        end

        local ar = target:tool("ar") or "ar"
        local targetfile = target:targetfile()
        os.mkdir(path.directory(targetfile))

        local ar_args = {"-cr", targetfile}
        for _, objectfile in ipairs(objectfiles) do
            table.insert(ar_args, objectfile)
        end
        os.vrunv(ar, ar_args)
    end)

    add_includedirs("../include", "../src")

    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    -- Link common runtime library names shipped by MACA.
    add_syslinks("mcruntime", "mxc-runtime64", "runtime_cu", {public = true})

    on_install(function (target) end)
target_end()
