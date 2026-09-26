cmake_minimum_required(VERSION 3.30 FATAL_ERROR)

# ---------------------------------------------------------------------------------------
# Project
# ---------------------------------------------------------------------------------------
project(sgl
    VERSION 0.1.0
    DESCRIPTION "C++20 module SIMD accelerated computational geometry library"
    LANGUAGES CXX
)

# ---------------------------------------------------------------------------------------
# C++ standard (modules require C++20 or newer)
# ---------------------------------------------------------------------------------------
if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 20)
endif()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_SCAN_FOR_MODULES ON)

if(CMAKE_CXX_STANDARD LESS 20)
    message(FATAL_ERROR
        "sgl: CMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD} is too low"
        "C++20 modules require C++20 or newer.")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------------------
# Compiler support check: each minimum is the higher of the C++20 named-module floor
# and the CWG 2518 floor.
# ---------------------------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "17")
        message(FATAL_ERROR "sgl: Clang >= 17 required (C++20 modules + CWG 2518) "
            "(found ${CMAKE_CXX_COMPILER_VERSION}).")
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "14")
        message(FATAL_ERROR "sgl: GCC >= 14 required (C++20 modules; CWG 2518 since 13) "
            "(found ${CMAKE_CXX_COMPILER_VERSION}).")
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "19.40")
        message(FATAL_ERROR "sgl: MSVC >= 19.40 (VS 2022 17.10) required for CWG 2518 "
            "(found ${CMAKE_CXX_COMPILER_VERSION}).")
    endif()
else()
    message(WARNING "sgl: untested compiler '${CMAKE_CXX_COMPILER_ID}'. "
        "Supported toolchains: GCC >= 14, Clang >= 17, MSVC >= 19.40.")
endif()

# ---------------------------------------------------------------------------------------
# Arch
# ---------------------------------------------------------------------------------------
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _PROC_LOWER)

if (_PROC_LOWER MATCHES "^(x86_64|amd64)$")
    set(IS_X86_64 TRUE)
    set(IS_ARM64 FALSE)
elseif (_PROC_LOWER MATCHES "^(aarch64|arm64)$")
    set(IS_X86_64 FALSE)
    set(IS_ARM64 TRUE)
else()
    message(FATAL_ERROR "sgl: arch ${CMAKE_SYSTEM_PROCESSOR} not supported")
endif()

# ---------------------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------------------
option(SGL_PIC   "Build with position-independent code" OFF)
option(SGL_TESTS "Build sgl tests" OFF)
option(SGL_BENCHMARKS "Build sgl benchmarks" OFF)
option(SGL_INSTALL "Generate install + find_package() rules" "${PROJECT_IS_TOP_LEVEL}")

# ---------------------------------------------------------------------------------------
# Library (always STATIC: the module interface is the real deliverable)
# ---------------------------------------------------------------------------------------
add_library(${PROJECT_NAME} STATIC)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)

# PUBLIC: consumers recompile the inline module interface and need the same ISA.
# Keyed on the frontend variant, not the compiler id, so clang-cl gets /arch:AVX2.
# MSVC folds FMA3 into /arch:AVX2; there is no separate /arch:FMA.
target_compile_options(${PROJECT_NAME} PUBLIC
    "$<$<AND:$<BOOL:${IS_X86_64}>,$<CXX_COMPILER_FRONTEND_VARIANT:GNU>>:-mavx2;-mfma>"
    "$<$<AND:$<BOOL:${IS_X86_64}>,$<CXX_COMPILER_FRONTEND_VARIANT:MSVC>>:/arch:AVX2>"
)

if(SGL_PIC)
    set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# ---------------------------------------------------------------------------------------
# Compiler warnings
# ---------------------------------------------------------------------------------------
add_library(sgl_project_flags INTERFACE)

# MSVC is true for cl.exe and for clang-cl; both take the /-style warning flags.
if(MSVC)
    target_compile_options(sgl_project_flags INTERFACE
        /W4
        /permissive- # conformance mode (also wanted by the modules front-end)
        /WX
    )
else()
    target_compile_options(sgl_project_flags INTERFACE
        -Wall
        -Wpedantic
        -Wextra
        -Wshadow
        -Wconversion
        -Wnarrowing
        -Wfloat-conversion
        -Wsign-conversion
        -Wold-style-cast
        -Wundef
        -Wuninitialized
        # off: partial aggregate init of a vector or matrix value-initialises the rest, which
        # is the intent, not an oversight.
        -Wno-missing-field-initializers
        -Werror
    )

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        # Both TU-local-entity spellings are listed, so -Wno-unknown-warning-option
        # must come first: the name this Clang lacks must not become a -Werror.
        target_compile_options(sgl_project_flags INTERFACE
            -Wmove
            -Wno-unknown-warning-option
            -Wno-reference-tu-local-entity-in-other-tu  # Clang <= 21
            -Wno-TU-local-entity-exposure               # Clang 22+
        )
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(sgl_project_flags INTERFACE
            -Wpessimizing-move
            -Wredundant-move
        )
    endif()
endif()

# Top-level only: never impose -Werror on a consumer's build. BUILD_INTERFACE covers
# the installed export; the guard covers add_subdirectory, where it is still active.
if(PROJECT_IS_TOP_LEVEL)
    target_link_libraries(${PROJECT_NAME} PRIVATE $<BUILD_INTERFACE:sgl_project_flags>)
endif()

# ---------------------------------------------------------------------------------------
# Module interface units
# ---------------------------------------------------------------------------------------
set(ModuleInterfaceFiles
    src/sgl/types.cppm
    src/sgl/scalar.cppm
    src/sgl/sgl.cppm
)

if(IS_X86_64)
    list(APPEND ModuleInterfaceFiles src/sgl/avx.cppm)
elseif(IS_ARM64)
    list(APPEND ModuleInterfaceFiles src/sgl/neon.cppm)
endif()

target_sources(${PROJECT_NAME}
  PUBLIC
  FILE_SET sgl_cxx TYPE CXX_MODULES
      BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
      FILES ${ModuleInterfaceFiles}
)

# ---------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------
# Top-level only: `unit_tests` would collide with a consumer's own target.
if(PROJECT_IS_TOP_LEVEL AND SGL_TESTS)
    enable_testing() # at the top level so `ctest --test-dir build` is the CTest root
    add_subdirectory(tests)
endif()

if(PROJECT_IS_TOP_LEVEL AND SGL_BENCHMARKS)
    add_subdirectory(bench)
endif()

# ---------------------------------------------------------------------------------------
# Install + find_package(sgl) support
# ---------------------------------------------------------------------------------------
if(SGL_INSTALL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    set(_sgl_moddir ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}/cxxmodules)

    # cmake-lint: disable=E1122
    # ARCHIVE and FILE_SET are separate artifact groups, not a duplicate DESTINATION.
    install(TARGETS ${PROJECT_NAME}
        EXPORT ${PROJECT_NAME}-targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        FILE_SET sgl_cxx DESTINATION ${_sgl_moddir}
    )

    install(EXPORT ${PROJECT_NAME}-targets
        NAMESPACE ${PROJECT_NAME}::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
        FILE ${PROJECT_NAME}-targets.cmake
        CXX_MODULES_DIRECTORY cxx-modules
    )

    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}-config.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
    )

    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )

    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
    )
endif()
