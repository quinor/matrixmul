cmake_minimum_required(VERSION 2.8)
project(matrixmul)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_definitions("-std=c++17")

set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})

include_directories(MPI_INCLUDE_PATH)
include_directories(src)

add_executable(
    matrixmul
    src/main.cc
    src/load.cc
    src/column.cc
    src/inner.cc
    densematgen.cpp
)

set_target_properties(
    matrixmul PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/
)

target_link_libraries(
    matrixmul
    ${MPI_LIBRARIES}
    ${OpenMP_CXX_LIBRARIES}
)
