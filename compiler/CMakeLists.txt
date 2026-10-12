cmake_minimum_required(VERSION 3.16)
project(optifine C)

# Strict ISO C11: no GNU extensions, so the tree cannot silently depend on
# POSIX names such as M_PI (the sources that need it define a fallback) or on
# gnu11-only syntax.
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# Off by default so a newer compiler's new warning does not break a user's
# build; CI turns it on to keep the tree warning-clean.
option(OPTIFINE_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

# Settings every target in this tree shares: the public include directory,
# warnings, and the platform definitions. Production and test code both link
# it, so they are compiled under the same rules.
add_library(optifine_build_settings INTERFACE)
target_include_directories(optifine_build_settings INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(optifine_build_settings INTERFACE -Wall -Wextra -Wpedantic)
    if(OPTIFINE_WARNINGS_AS_ERRORS)
        target_compile_options(optifine_build_settings INTERFACE -Werror)
    endif()
endif()
if(MSVC OR CMAKE_C_SIMULATE_ID STREQUAL "MSVC")
    target_compile_definitions(optifine_build_settings INTERFACE _CRT_SECURE_NO_WARNINGS)
    if(OPTIFINE_WARNINGS_AS_ERRORS)
        target_compile_options(optifine_build_settings INTERFACE /WX)
    endif()
endif()

# The generic AVR backend: MIR, its verifier, AVR selection from MIR,
# instruction pricing and assembly emission. It knows nothing of the workload
# graph (ir.h) or its lowering, and a frontend that builds MIR links this
# alone (tests/test_backend_link checks that it stays so).
add_library(optifine_backend STATIC
    src/cost_model.c
    src/emit.c
    src/invariant.c
    src/mir/mir.c
    src/mir/mir_verify.c
    src/codegen/avr_instr.c
    src/codegen/avr_mir.c
    src/codegen/avr_mir_program.c
    src/codegen/cost_category.c
    src/codegen/instr_buf.c
)
target_link_libraries(optifine_backend PUBLIC optifine_build_settings)

# The workload compiler, everything except the command-line driver: the ONNX
# and DSP frontends, the workload graph, its lowering and translation to MIR,
# and program assembly, on top of the backend. A static library rather than
# an object library: a test that substitutes one of its own definitions for a
# library function (test_periodic) links only the archive members it actually
# needs.
add_library(optifine_core STATIC
    src/ir.c
    src/ir_verify.c
    src/pb_reader.c
    src/ingest.c
    src/dsp_build.c
    src/codegen/candidates.c
    src/codegen/dsp32.c
    src/codegen/hir_to_mir.c
    src/codegen/lower.c
    src/codegen/lower_dsp.c
    src/codegen/lower_fft.c
    src/codegen/lower_magnitude.c
    src/codegen/lower_ml.c
    src/codegen/lower_peak.c
    src/codegen/lower_requantize.c
    src/codegen/periodic.c
    src/codegen/program.c
    src/codegen/reuse_analysis.c
    src/codegen/registers.c
    src/codegen/select.c
    src/codegen/sram_layout.c
    src/frontend/c_frontend.c
)
target_link_libraries(optifine_core PUBLIC optifine_backend)
if(UNIX)
    # glibc/musl keep cos/lround/llround in libm; MSVC's CRT provides them
    # without a separate library.
    target_link_libraries(optifine_core PUBLIC m)
endif()

add_executable(optifine src/main.c)
target_link_libraries(optifine PRIVATE optifine_core)

enable_testing()
#add_subdirectory(tests)
