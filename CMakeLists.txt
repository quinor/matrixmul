cmake_minimum_required(VERSION 2.8)
project(matrixmul)

find_package(Boost COMPONENTS program_options REQUIRED)
find_package(MPI REQUIRED)

add_definitions("-std=c++17")

set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})

include_directories(MPI_INCLUDE_PATH)
include_directories(src)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

add_executable(
    matrixmul
    src/main.cc
    src/load.cc
    src/inner.cc
    src/column.cc
    src/densematgen.cpp
)

set_target_properties(
    matrixmul PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/
)

target_link_libraries(
    matrixmul
    ${Boost_LIBRARIES}
    ${MPI_LIBRARIES}
)