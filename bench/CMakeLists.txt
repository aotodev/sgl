include(FetchContent)
FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark
    GIT_TAG v1.9.5
    SYSTEM
)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(
    googlebenchmark
)

add_executable(soa_bench src/soa_bench.cpp)
target_link_libraries(soa_bench PRIVATE sgl sgl_project_flags benchmark::benchmark_main)
# The branchless baselines use & on bools on purpose. A source property, so it lands after -Wall.
set_source_files_properties(src/soa_bench.cpp PROPERTIES
    COMPILE_OPTIONS $<$<CXX_COMPILER_ID:Clang,AppleClang>:-Wno-bitwise-instead-of-logical>
)
