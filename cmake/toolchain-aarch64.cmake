# ---------------------------------------------------------------------------------------
# Cross-compile sgl to aarch64 (the NEON path), tested under qemu-user.
# ---------------------------------------------------------------------------------------
set(CMAKE_SYSTEM_NAME      Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(_sgl_prefix aarch64-linux-gnu-)

find_program(_sgl_aarch64_cxx ${_sgl_prefix}g++ NO_CACHE)
if(NOT _sgl_aarch64_cxx)
    message(FATAL_ERROR
        "sgl: ${_sgl_prefix}g++ not found. Install the aarch64-linux-gnu toolchain.")
endif()

set(CMAKE_C_COMPILER   ${_sgl_prefix}gcc)
set(CMAKE_CXX_COMPILER ${_sgl_prefix}g++)
set(CMAKE_AR           ${_sgl_prefix}ar)
set(CMAKE_RANLIB       ${_sgl_prefix}ranlib)
set(CMAKE_LINKER_TYPE  BFD)

execute_process(
    COMMAND ${CMAKE_CXX_COMPILER} -print-sysroot
    OUTPUT_VARIABLE _sgl_sysroot
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT _sgl_sysroot OR _sgl_sysroot STREQUAL "/")
    set(_sgl_sysroot /usr/aarch64-linux-gnu)
endif()

set(CMAKE_SYSROOT        ${_sgl_sysroot})
set(CMAKE_FIND_ROOT_PATH ${_sgl_sysroot})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Lets ctest and gtest_discover_tests run the cross-built binaries directly. Only set
# when qemu is present, so a build-only environment still configures.
find_program(_sgl_qemu qemu-aarch64 NO_CACHE)
if(_sgl_qemu)
    # -cpu max: keeps qemu from capping which intrinsics the library may use.
    set(CMAKE_CROSSCOMPILING_EMULATOR ${_sgl_qemu};-cpu;max;-L;${_sgl_sysroot})
endif()
