# Consumes an installed sgl through find_package, as a downstream project would.
cmake_minimum_required(VERSION 3.30)
project(sgl_install_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(sgl REQUIRED)

add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE sgl::sgl)

enable_testing()
add_test(NAME consumer COMMAND consumer)
