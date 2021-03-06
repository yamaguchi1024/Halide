if (NOT CMAKE_GENERATOR MATCHES "Make|Ninja")
    message(STATUS "Notice: ${CMAKE_GENERATOR} does not support depfile dependencies. Incremental builds may fail.")
endif ()

# Keep these lists in alphabetical order.
set(RUNTIME_CPP
    aarch64_cpu_features
    alignment_128
    alignment_32
    alignment_64
    allocation_cache
    android_clock
    android_host_cpu_count
    android_io
    arm_cpu_features
    cache
    can_use_target
    cuda
    destructors
    device_interface
    errors
    fake_get_symbol
    fake_thread_pool
    float16_t
    fuchsia_clock
    fuchsia_host_cpu_count
    fuchsia_yield
    gpu_device_selection
    halide_buffer_t
    hexagon_cache_allocator
    hexagon_cpu_features
    hexagon_dma
    hexagon_dma_pool
    hexagon_host
    ios_io
    linux_clock
    linux_host_cpu_count
    linux_yield
    matlab
    metadata
    metal
    metal_objc_arm
    metal_objc_x86
    mips_cpu_features
    module_aot_ref_count
    module_jit_ref_count
    msan
    msan_stubs
    opencl
    opengl
    opengl_egl_context
    opengl_glx_context
    openglcompute
    osx_clock
    osx_get_symbol
    osx_host_cpu_count
    osx_opengl_context
    osx_yield
    posix_abort
    posix_allocator
    posix_clock
    posix_error_handler
    posix_get_symbol
    posix_io
    posix_print
    posix_threads
    posix_threads_tsan
    powerpc_cpu_features
    prefetch
    profiler
    profiler_inlined
    pseudostack
    qurt_allocator
    qurt_hvx
    qurt_hvx_vtcm
    qurt_init_fini
    qurt_threads
    qurt_threads_tsan
    qurt_yield
    riscv_cpu_features
    runtime_api
    ssp
    to_string
    trace_helper
    tracing
    wasm_cpu_features
    windows_abort
    windows_clock
    windows_cuda
    windows_d3d12compute_x86
    windows_get_symbol
    windows_io
    windows_opencl
    windows_profiler
    windows_threads
    windows_threads_tsan
    windows_yield
    write_debug_image
    x86_cpu_features
    )

set(RUNTIME_LL
    aarch64
    arm
    arm_no_neon
    hvx_128
    hvx_64
    mips
    posix_math
    powerpc
    ptx_dev
    wasm_math
    win32_math
    x86
    x86_avx
    x86_avx2
    x86_sse41
    )

set(RUNTIME_BC
    compute_20
    compute_30
    compute_35
    )

set(RUNTIME_HEADER_FILES
    HalideBuffer.h
    HalidePyTorchCudaHelpers.h
    HalidePyTorchHelpers.h
    HalideRuntime.h
    HalideRuntimeCuda.h
    HalideRuntimeD3D12Compute.h
    HalideRuntimeHexagonDma.h
    HalideRuntimeHexagonHost.h
    HalideRuntimeMetal.h
    HalideRuntimeOpenCL.h
    HalideRuntimeOpenGL.h
    HalideRuntimeOpenGLCompute.h
    HalideRuntimeQurt.h
    )

# Need to create an object library for this because CMake
# doesn't support using target_sources on a target declared
# in a different directory ONLY IF that source was created
# by add_custom_command, as is the case in this directory.
add_library(Halide_initmod OBJECT)

# Note: ensure that these flags match the flags in the Makefile.
# Note: this always uses Clang-from-LLVM for compilation, so none of these flags should need conditionalization.
set(RUNTIME_CXX_FLAGS
    -O3
    -ffreestanding
    -fno-blocks
    -fno-exceptions
    -fno-unwind-tables
    -fno-vectorize
    # Note: we don't want static locals to get thread synchronization stuff.
    -fno-threadsafe-statics
    -Wall
    -Wcast-qual
    -Werror
    -Wignored-qualifiers
    -Wno-comment
    -Wno-psabi
    -Wno-unknown-warning-option
    -Wno-unused-function
    -Wsign-compare
)

foreach (i IN LISTS RUNTIME_CPP)
    foreach (j IN ITEMS 32 64)
        # -fpic needs special treatment; see below on windows 64bits
        set(fpic -fpic)
        if (j EQUAL 32)
            if (i MATCHES "windows_.*")
                # win32 uses the stdcall calling convention, which is x86-specific
                set(TARGET "i386-unknown-unknown-unknown")
            else ()
                # (The 'nacl' is a red herring. This is just a generic 32-bit little-endian target.)
                set(TARGET "le32-unknown-nacl-unknown")
            endif ()
        else ()
            if (i MATCHES "windows_.*")
                # must omit -fpic, otherwise clang will complain with the following:
                # clang : error : unsupported option '-fpic' for target 'x86_64-pc-windows-msvc'
                set(fpic "")
                set(TARGET "x86_64-unknown-windows-unknown")
            else ()
                # generic 64-bit code
                set(TARGET "le64-unknown-unknown-unknown")
            endif ()
        endif ()

        set(SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/${i}.cpp")

        set(RUNTIME_DEFINES -DCOMPILING_HALIDE_RUNTIME -DBITS_${j})
        set(RUNTIME_DEFINES_debug -g -DDEBUG_RUNTIME ${RUNTIME_DEFINES})

        foreach (SUFFIX IN ITEMS "" "_debug")
            set(basename "initmod.${i}_${j}${SUFFIX}")
            set(LL "${basename}.ll")
            set(BC "${basename}.bc")
            set(INITMOD "_initmod_${i}_${j}${SUFFIX}.cpp")
            set(SYMBOL "halide_internal_initmod_${i}_${j}${SUFFIX}")

            set(clang_flags ${RUNTIME_CXX_FLAGS} ${fpic} ${RUNTIME_DEFINES${SUFFIX}} -m${j} -target ${TARGET} -emit-llvm -S)

            # Dep-files are subtle and require clang to run using *just* the right
            # relative paths to the build root, NOT the Halide build root. This is
            # a perfect storm of bad behavior from CMake, Ninja, and Clang.
            file(RELATIVE_PATH ll_path "${CMAKE_BINARY_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/${LL}")
            file(TO_NATIVE_PATH "${ll_path}" ll_path)

            if (CMAKE_GENERATOR MATCHES "Ninja")
                list(APPEND clang_flags -MD -MF "$<SHELL_PATH:${CMAKE_CURRENT_BINARY_DIR}/${basename}.d>")
                set(dep_args DEPFILE "${CMAKE_CURRENT_BINARY_DIR}/${basename}.d")
            elseif (NOT CMAKE_GENERATOR MATCHES "Make")
                set(dep_args "")
            endif ()

            add_custom_command(OUTPUT "${LL}"
                               COMMAND clang ${clang_flags} -o "${ll_path}" "$<SHELL_PATH:${SOURCE}>"
                               DEPENDS "${SOURCE}"
                               # Note: IMPLICIT_DEPENDS only works for Makefile generators, ${dep_args} handles Ninja.
                               IMPLICIT_DEPENDS CXX "${SOURCE}"
                               WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
                               ${dep_args}
                               VERBATIM)

            add_custom_command(OUTPUT "${BC}"
                               COMMAND llvm-as "${LL}" -o "${BC}"
                               DEPENDS "${LL}"
                               VERBATIM)

            add_custom_command(OUTPUT "${INITMOD}"
                               COMMAND binary2cpp ${SYMBOL} < "${BC}" > "${INITMOD}"
                               DEPENDS "${BC}" binary2cpp
                               VERBATIM)

            target_sources(Halide_initmod PRIVATE ${INITMOD})
        endforeach ()
    endforeach ()
endforeach ()


foreach (i IN LISTS RUNTIME_LL)
    set(LL "${i}.ll")
    set(BC "initmod.${i}.bc")
    set(INITMOD "_initmod_${i}.cpp")

    add_custom_command(OUTPUT "${BC}"
                       COMMAND llvm-as "$<SHELL_PATH:${CMAKE_CURRENT_SOURCE_DIR}/${LL}>" -o "${BC}"
                       DEPENDS "${LL}"
                       VERBATIM)
    add_custom_command(OUTPUT "${INITMOD}"
                       COMMAND binary2cpp "halide_internal_initmod_${i}_ll" < "${BC}" > "${INITMOD}"
                       DEPENDS "${BC}" binary2cpp
                       VERBATIM)
    target_sources(Halide_initmod PRIVATE ${INITMOD})
endforeach ()

foreach (i IN LISTS RUNTIME_BC)
    set(INITMOD "_initmod_ptx_${i}.cpp")
    set(RT_BC "${CMAKE_CURRENT_SOURCE_DIR}/nvidia_libdevice_bitcode/libdevice.${i}.10.bc")

    add_custom_command(OUTPUT "${INITMOD}"
                       COMMAND binary2cpp "halide_internal_initmod_ptx_${i}_ll" < "$<SHELL_PATH:${RT_BC}>" > "${INITMOD}"
                       DEPENDS binary2cpp "${RT_BC}"
                       VERBATIM)
    target_sources(Halide_initmod PRIVATE ${INITMOD})
endforeach ()

add_custom_command(OUTPUT "_initmod_inlined_c.cpp"
                   COMMAND binary2cpp "halide_internal_initmod_inlined_c" < "$<SHELL_PATH:${CMAKE_CURRENT_SOURCE_DIR}/halide_buffer_t.cpp>" > "_initmod_inlined_c.cpp"
                   DEPENDS "halide_buffer_t.cpp" binary2cpp
                   VERBATIM)
target_sources(Halide_initmod PRIVATE "_initmod_inlined_c.cpp")

foreach (i IN LISTS RUNTIME_HEADER_FILES)
    string(REPLACE "." "_" SYM_NAME "${i}")
    add_custom_command(OUTPUT "_initmod_${SYM_NAME}.cpp"
                       COMMAND binary2cpp "halide_internal_runtime_header_${SYM_NAME}" < "$<SHELL_PATH:${CMAKE_CURRENT_SOURCE_DIR}/${i}>" > "_initmod_${SYM_NAME}.cpp"
                       DEPENDS "${i}" binary2cpp
                       VERBATIM)
    target_sources(Halide_initmod PRIVATE "_initmod_${SYM_NAME}.cpp")

    configure_file(${i} "${Halide_BINARY_DIR}/include/${i}" COPYONLY)
endforeach ()

##
# Target for the runtime
##

add_library(Halide_Runtime INTERFACE)
add_library(Halide::Runtime ALIAS Halide_Runtime)
target_include_directories(Halide_Runtime INTERFACE $<BUILD_INTERFACE:${Halide_BINARY_DIR}/include>)
set_target_properties(Halide_Runtime PROPERTIES EXPORT_NAME Runtime)

