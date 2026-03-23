-- Iluvatar CoreX GPU targets
-- Uses clang++ with CUDA frontend, NOT nvcc
-- We use on_build to completely bypass xmake's CUDA toolchain detection

target("llaisys-device-iluvatar")
    set_kind("static")
    add_deps("llaisys-utils")
    set_languages("cxx17")
    set_warnings("all", "error")

    -- Do NOT add .cu files via add_files - that triggers xmake CUDA toolchain
    -- Instead, build everything in on_build
    on_build(function (target)
        import("core.project.depend")

        local sourcedir = path.absolute("src/device/iluvatar")
        local sources = {
            path.join(sourcedir, "iluvatar_runtime_api.cu"),
            path.join(sourcedir, "iluvatar_resource.cu"),
        }

        local objectfiles = {}
        for _, sourcefile in ipairs(sources) do
            local objectfile = target:objectfile(sourcefile)
            local objectdir = path.directory(objectfile)
            if not os.isdir(objectdir) then
                os.mkdir(objectdir)
            end

            local dependfile = target:dependfile(objectfile)
            depend.on_changed(function ()
                local argv = {
                    "-x", "cuda",
                    "--cuda-gpu-arch=ivcore10",
                    "--cuda-path=/usr/local/corex",
                    "-std=c++17",
                    "-fPIC",
                    "-O3",
                    "-DENABLE_ILUVATAR_API",
                    "-Iinclude",
                    "-I/usr/local/corex/include",
                    "-c",
                    "-o", objectfile,
                    sourcefile
                }
                os.vrunv("/usr/local/corex/bin/clang++", argv)
            end, {dependfile = dependfile, files = {sourcefile}})

            table.insert(objectfiles, objectfile)
        end

        -- Archive into static library
        local targetfile = target:targetfile()
        local targetdir = path.directory(targetfile)
        if not os.isdir(targetdir) then
            os.mkdir(targetdir)
        end
        os.vrunv("ar", {"-cr", targetfile, table.unpack(objectfiles)})
    end)

    on_install(function (target) end)
target_end()

target("llaisys-ops-iluvatar")
    set_kind("static")
    add_deps("llaisys-tensor")
    set_languages("cxx17")
    set_warnings("all", "error")

    -- Do NOT add .cu files via add_files
    on_build(function (target)
        import("core.project.depend")

        -- Find all .cu files under src/ops/*/nvidia/
        local sources = os.files("src/ops/*/nvidia/*.cu")

        local objectfiles = {}
        for _, sourcefile in ipairs(sources) do
            local objectfile = target:objectfile(sourcefile)
            local objectdir = path.directory(objectfile)
            if not os.isdir(objectdir) then
                os.mkdir(objectdir)
            end

            local dependfile = target:dependfile(objectfile)
            depend.on_changed(function ()
                local argv = {
                    "-x", "cuda",
                    "--cuda-gpu-arch=ivcore10",
                    "--cuda-path=/usr/local/corex",
                    "-std=c++17",
                    "-fPIC",
                    "-O3",
                    "-DENABLE_ILUVATAR_API",
                    "-Iinclude",
                    "-I/usr/local/corex/include",
                    "-c",
                    "-o", objectfile,
                    sourcefile
                }
                os.vrunv("/usr/local/corex/bin/clang++", argv)
            end, {dependfile = dependfile, files = {sourcefile}})

            table.insert(objectfiles, objectfile)
        end

        -- Archive into static library
        local targetfile = target:targetfile()
        local targetdir = path.directory(targetfile)
        if not os.isdir(targetdir) then
            os.mkdir(targetdir)
        end
        os.vrunv("ar", {"-cr", targetfile, table.unpack(objectfiles)})
    end)

    on_install(function (target) end)
target_end()
