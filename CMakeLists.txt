cmake_minimum_required(VERSION 2.8)
project(matrixmul)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(MKL REQUIRED)


add_definitions("-std=c++14")

set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${OpenMP_CXX_FLAGS} ${MPI_COMPILE_FLAGS})
set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${OpenMP_CXX_LIBRARIES} ${MPI_LINK_FLAGS})

include_directories(${MPI_INCLUDE_PATH})
include_directories(${MKL_INCLUDE_DIR})
include_directories(src)

link_directories(${MKL_LIBRARY_DIR})
link_directories(${MKL_LIBRARIES})

add_executable(
    matrixmul
    src/main.cc
    src/load.cc
    src/column.cc
    src/inner.cc
    densematgen.cpp
)

# set_target_properties(
#     matrixmul PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/
# )

target_link_libraries(
    matrixmul
    ${MPI_LIBRARIES}
    ${OpenMP_CXX_LIBRARIES}
    mkl_intel_lp64
    mkl_sequential
    mkl_core
)
