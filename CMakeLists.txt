cmake_minimum_required(VERSION 3.21)

project(PARADYMS)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Kokkos REQUIRED)

file(GLOB SOURCES src/*.cpp)

add_executable(paradyms src/main.cpp)
add_executable(dyngraph src/generate_graph.cpp)

target_include_directories(paradyms PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(paradyms PRIVATE Kokkos::kokkos)
if(EXECSPACE)
    string(TOUPPER "${EXECSPACE}" MY_EXECSPACE)
    if(MY_EXECSPACE STREQUAL "OPENMP")
        add_compile_definitions(EXECSPACE=2)
        message("-- " EXECSPACE="${MY_EXECSPACE}")
    elseif(MY_EXECSPACE STREQUAL "CUDA")
        add_compile_definitions(EXECSPACE=1)
        message("-- " EXECSPACE="${MY_EXECSPACE}")
    else()
        add_compile_definitions(EXECSPACE=2)
    endif()
endif()

# TODO: get add_compile_options to work
# add_compile_options(-ffast-math -funroll-loops -fno-strict-aliasing -fomit-frame-pointer -fexpensive-optimizations -funroll-loops -fmove-loop-invariants -fprefetch-loop-arrays -ftree-loop-optimize -ftree-vect-loop-version -ftree-vectorize)
# add_link_options(-ffast-math -funroll-loops -fno-strict-aliasing -fomit-frame-pointer -fexpensive-optimizations -funroll-loops -fmove-loop-invariants -fprefetch-loop-arrays -ftree-loop-optimize -ftree-vect-loop-version -ftree-vectorize)

install(TARGETS paradyms dyngraph DESTINATION ${CMAKE_INSTALL_BINDIR})