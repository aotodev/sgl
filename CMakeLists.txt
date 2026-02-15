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
# Compiler support check
# ---------------------------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "16")
        message(FATAL_ERROR "sgl: Clang >= 16 required for C++20 modules "
            "(found ${CMAKE_CXX_COMPILER_VERSION}).")
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "14")
        message(FATAL_ERROR "sgl: GCC >= 14 required for C++20 modules "
            "(found ${CMAKE_CXX_COMPILER_VERSION}).")
    endif()
else()
    message(WARNING "sgl: untested compiler '${CMAKE_CXX_COMPILER_ID}'. "
        "Clang >= 16 or GCC >= 14 are the supported toolchains.")
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
option(SGL_INSTALL "Generate install + find_package() rules" "${PROJECT_IS_TOP_LEVEL}")

# ---------------------------------------------------------------------------------------
# Library (always STATIC)
#
# The real interface is the C++20 module, which every consumer recompiles from
# the shipped module sources against their own toolchain (BMIs are not portable).
# The compiled archive is nearly empty: the API is templates plus inline SIMD
# functions, all instantiated/inlined in the consumer's translation units. There
# is therefore no stable ABI and no meaningful binary payload to ship as a shared
# object, so a SHARED build would buy nothing.
# ---------------------------------------------------------------------------------------
add_library(${PROJECT_NAME} STATIC)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)

# AVX2 + FMA on x86-64. PUBLIC so the installed export carries it to consumers,
# whose recompilation of the (inline) module interface needs the same ISA.
target_compile_options(${PROJECT_NAME} PUBLIC "$<$<BOOL:${IS_X86_64}>:-mavx2;-mfma>")

if(SGL_PIC)
    set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# ---------------------------------------------------------------------------------------
# Compiler warnings
# ---------------------------------------------------------------------------------------
add_library(project_flags INTERFACE)

target_compile_options(project_flags INTERFACE
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
    -Werror
)

# Compiler-specific warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(project_flags INTERFACE
        -Wmove
        -Wno-reference-tu-local-entity-in-other-tu
    )
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(project_flags INTERFACE
        -Wpessimizing-move
        -Wredundant-move
    )
endif()

# BUILD_INTERFACE: project flags/warnings are a build-only concern and must
# not leak into the installed export or onto consumers.
target_link_libraries(${PROJECT_NAME} PRIVATE $<BUILD_INTERFACE:project_flags>)

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
if(SGL_TESTS)
    add_subdirectory(tests)
endif()

# ---------------------------------------------------------------------------------------
# Install + find_package(sgl) support
#
# Installs the arch-selected module source (consumers recompile the BMI with a
# compatible toolchain), the static archive, and a package config. The exported
# target carries the AVX2/FMA usage requirement so importers compile the module
# interface with the matching ISA. add_subdirectory / FetchContent remains the
# most robust way to consume a module library.
# ---------------------------------------------------------------------------------------
if(SGL_INSTALL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    set(_sgl_moddir ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}/cxxmodules)

    # cmake-lint: disable=E1122
    # (false positive: ARCHIVE and FILE_SET are separate install(TARGETS)
    # artifact groups, each with its own DESTINATION, not a duplicate keyword)
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
