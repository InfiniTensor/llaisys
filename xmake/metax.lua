-- MetaX MACA backend integration.

local function add_unique(list, value)
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

local function add_if_dir(list, dir)
    if os.isdir(dir) then
        add_unique(list, dir)
    end
end

local function sorted_files(pattern)
    local files = os.files(pattern)
    table.sort(files)
    return files
end

local function build_metax_config()
    local projectdir = os.projectdir()
    local maca_path = get_config("maca-path") or os.getenv("MACA_HOME") or "/opt/maca"
    local mxdriver_path = get_config("mxdriver-path") or "/opt/mxdriver"

    local roots = {}
    add_unique(roots, maca_path)
    add_unique(roots, os.getenv("MACA_HOME"))
    add_unique(roots, "/opt/maca")
    add_unique(roots, "/usr/local/maca")
    add_unique(roots, "/opt/maca-3.3.0")
    add_unique(roots, "/opt/maca-3.2.1")
    add_unique(roots, "/opt/maca-3.2.0")
    add_unique(roots, "/opt/maca-3.1.0")

    local include_dirs = {
        path.join(projectdir, "include"),
        path.join(projectdir, "src")
    }
    local link_dirs = {}
    for _, root in ipairs(roots) do
        add_if_dir(include_dirs, path.join(root, "include"))
        add_if_dir(include_dirs, path.join(root, "tools", "cu-bridge", "include"))
        add_if_dir(include_dirs, path.join(root, "mxgpu_llvm", "include"))

        add_if_dir(link_dirs, path.join(root, "lib"))
        add_if_dir(link_dirs, path.join(root, "lib64"))
        add_if_dir(link_dirs, path.join(root, "mxgpu_llvm", "lib"))
        add_if_dir(link_dirs, path.join(root, "mxgpu_llvm", "lib64"))
    end
    add_if_dir(link_dirs, path.join(mxdriver_path, "lib"))

    local mxcc = os.getenv("MXCC")
    if not mxcc or mxcc == "" then
        local candidate = path.join(maca_path, "mxgpu_llvm", "bin", "mxcc")
        mxcc = os.isfile(candidate) and candidate or "mxcc"
    end

    return {
        projectdir = projectdir,
        maca_path = maca_path,
        mxdriver_path = mxdriver_path,
        mxcc = mxcc,
        include_dirs = include_dirs,
        link_dirs = link_dirs,
        syslinks = {"mcruntime", "mxc-runtime64", "runtime_cu", "mxsml", "mcblas"},
        common_cxflags = is_plat("windows") and {} or {"-fPIC", "-Wno-unknown-pragmas"},
        maca_compile_flags = {
            "-std=c++17",
            "-D__CUDACC__",
            "-x", "maca",
            "-offload-arch", "native",
            "--maca-path=" .. maca_path,
            "-O3",
            "-fPIC",
            "-Wno-unknown-pragmas",
            "-DENABLE_METAX_API"
        }
    }
end

local function configure_current_target(cfg)
    set_languages("cxx17")
    set_warnings("all", "error")
    for _, flag in ipairs(cfg.common_cxflags) do
        add_cxflags(flag)
    end
    for _, includedir in ipairs(cfg.include_dirs) do
        add_includedirs(includedir, {public = true})
    end
    for _, linkdir in ipairs(cfg.link_dirs) do
        add_linkdirs(linkdir, {public = true})
    end
    add_syslinks(table.unpack(cfg.syslinks), {public = true})
end

local function object_path(cfg, group_name, source)
    local basename = path.basename(source) .. ".o"
    if group_name == "ops" then
        local op_name = path.basename(path.directory(path.directory(source)))
        return path.join(cfg.projectdir, "build", "_gen", "metax", group_name, op_name, basename)
    end
    return path.join(cfg.projectdir, "build", "_gen", "metax", group_name, basename)
end

local function emit_compile_commands(batchcmds, cfg, sources, objects)
    for i, source in ipairs(sources) do
        local object = objects[i]
        batchcmds:mkdir(path.directory(object))

        local args = {}
        for _, flag in ipairs(cfg.maca_compile_flags) do
            table.insert(args, flag)
        end
        for _, includedir in ipairs(cfg.include_dirs) do
            table.insert(args, "-I" .. includedir)
        end
        table.insert(args, "-c")
        table.insert(args, source)
        table.insert(args, "-o")
        table.insert(args, object)

        batchcmds:vrunv(cfg.mxcc, args)
    end
end

local function emit_archive_commands(batchcmds, target, objects)
    local targetfile = target:targetfile()
    local ar = target:tool("ar") or "ar"

    batchcmds:mkdir(path.directory(targetfile))
    batchcmds:rm(targetfile)

    local args = {"-cr", targetfile}
    for _, object in ipairs(objects) do
        table.insert(args, object)
    end
    batchcmds:vrunv(ar, args)
end

local function set_target_sources(target, cfg, group_name, pattern)
    local sources = sorted_files(path.join(cfg.projectdir, pattern))
    local objects = {}
    for _, source in ipairs(sources) do
        table.insert(objects, object_path(cfg, group_name, source))
    end
    target:data_set("metax_sources", sources)
    target:data_set("metax_objects", objects)
end

local function build_target_sources(target, batchcmds, cfg)
    local sources = target:data("metax_sources") or {}
    local objects = target:data("metax_objects") or {}
    emit_compile_commands(batchcmds, cfg, sources, objects)
    emit_archive_commands(batchcmds, target, objects)
end

local function register_metax_archive_target(cfg, name, deps, group_name, pattern)
    target(name)
        set_kind("static")
        for _, dep in ipairs(deps) do
            add_deps(dep)
        end
        configure_current_target(cfg)

        on_load(function (target)
            set_target_sources(target, cfg, group_name, pattern)
        end)

        on_buildcmd(function (target, batchcmds, opt)
            build_target_sources(target, batchcmds, cfg)
        end)

        on_install(function (target) end)
    target_end()
end

local metax = build_metax_config()

register_metax_archive_target(
    metax,
    "llaisys-device-metax",
    {"llaisys-utils"},
    "device",
    "src/device/metax/*.maca"
)

register_metax_archive_target(
    metax,
    "llaisys-ops-metax",
    {"llaisys-tensor"},
    "ops",
    "src/ops/*/metax/*.maca"
)
