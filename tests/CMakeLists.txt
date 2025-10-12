message(STATUS "Configuring unit tests")

include(CTest)
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest
    GIT_TAG v1.15.2
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE) # don't leak gtest into sgl's install prefix

FetchContent_MakeAvailable(
    googletest
)

file(GLOB_RECURSE TEST_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
add_executable(unit_tests ${TEST_SOURCE})

target_link_libraries(unit_tests PRIVATE sgl project_flags gtest gtest_main)

include(GoogleTest)
gtest_discover_tests(unit_tests)
