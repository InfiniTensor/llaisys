local maca_root = os.getenv("MACA_HOME") or "/opt/maca"
local maca_include = os.getenv("MACA_INCLUDEDIR") or path.join(maca_root, "include")
local maca_lib = os.getenv("MACA_LIBDIR") or path.join(maca_root, "lib")
local mxdriver_root = os.getenv("MXDRIVER_ROOT") or "/opt/mxdriver"
local mxdriver_lib = os.getenv("MXDRIVER_LIBDIR") or path.join(mxdriver_root, "lib")
local mxcc = get_config("mxcc") or os.getenv("MXCC") or path.join(maca_root, "mxgpu_llvm", "bin", "mxcc")

local function _metax_objectfiles(target)
    local objectfiles = {}
    for _, sourcefile in ipairs(target:values("metax.files") or {}) do
        table.insert(objectfiles, {sourcefile = sourcefile, objectfile = target:objectfile(sourcefile)})
    end
    return objectfiles
end

local function _metax_compile_argv(target, sourcefile, objectfile)
    local argv = {
        "-x", "maca",
        "-std=c++17",
        "-c",
        path(sourcefile),
        "-o", path(objectfile)
    }

    if not is_plat("windows") then
        table.insert(argv, "-fPIC")
    end

    if target:get("symbols") == "debug" then
        table.insert(argv, "-g")
    end

    local optimize = target:get("optimize")
    if optimize == "none" then
        table.insert(argv, "-O0")
    elseif optimize == "fast" then
        table.insert(argv, "-O2")
    elseif optimize == "faster" or optimize == "fastest" then
        table.insert(argv, "-O3")
    elseif optimize == "smallest" then
        table.insert(argv, "-O1")
    end

    for _, define in ipairs(target:get("defines") or {}) do
        table.insert(argv, "-D" .. define)
    end
    for _, define in ipairs(target:get("undefines") or {}) do
        table.insert(argv, "-U" .. define)
    end
    for _, dir in ipairs(target:get("includedirs") or {}) do
        table.insert(argv, "-I" .. path.absolute(dir, os.projectdir()))
    end
    for _, dir in ipairs(target:get("sysincludedirs") or {}) do
        table.insert(argv, "-isystem")
        table.insert(argv, path.absolute(dir, os.projectdir()))
    end
    return argv
end

rule("metax")
    after_load(function (target)
        for _, item in ipairs(_metax_objectfiles(target)) do
            table.insert(target:objectfiles(), item.objectfile)
        end
    end)

    before_build(function (target, opt)
        import("core.project.depend")
        import("utils.progress")

        for index, item in ipairs(_metax_objectfiles(target)) do
            local sourcefile = item.sourcefile
            local objectfile = item.objectfile
            local dependfile = target:dependfile(objectfile)
            local dependinfo = target:is_rebuilt() and {} or (depend.load(dependfile) or {})
            if depend.is_changed(dependinfo, {lastmtime = os.mtime(objectfile)}) then
                progress.show((index * 100) / math.max(#(target:values("metax.files") or {}), 1),
                              "${color.build.object}compiling.metax %s", sourcefile)
                os.mkdir(path.directory(objectfile))
                os.vrunv(mxcc, _metax_compile_argv(target, sourcefile, objectfile))
                dependinfo.files = {sourcefile}
                depend.save(dependinfo, dependfile)
            end
        end
    end)

local function add_metax_common()
    set_languages("cxx17")
    set_warnings("all", "error")
    add_includedirs(maca_include)
    add_linkdirs(maca_lib, mxdriver_lib)
    add_rpathdirs(maca_lib, mxdriver_lib)
    add_rules("metax")
end

target("llaisys-device-metax")
    set_kind("static")
    add_metax_common()
    add_values("metax.files", os.files(path.join(os.projectdir(), "src/device/metax/*.cu")))

    on_install(function(target) end)
target_end()

target("llaisys-ops-metax")
    set_kind("static")
    add_deps("llaisys-tensor")
    add_deps("llaisys-device-metax")
    add_metax_common()
    add_values("metax.files", os.files(path.join(os.projectdir(), "src/ops/metax/*.cu")))

    on_install(function(target) end)
target_end()
