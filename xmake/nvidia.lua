function add_nvidia_build_settings()
    set_languages("cxx17")
    set_policy("build.cuda.devlink", true)
    add_cugencodes("native")
    add_cuflags("-Xcompiler=-fPIC")
    add_links("cudart", "cublas", "cudadevrt")
end

function add_nvidia_source_files()
    add_files("src/device/nvidia/*.cu")
    add_files("src/ops/*/nvidia/*.cu")
end
