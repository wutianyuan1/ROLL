cmake_minimum_required(VERSION 3.12)
project(global_scheduler)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11
find_package(pybind11 REQUIRED)

# Create weave simulator module
pybind11_add_module(global_scheduler_cpp
    weave_sim.cpp
    brute_force_solver.cpp
    bindings.cpp
)

# Optimize for performance
target_compile_options(global_scheduler_cpp PRIVATE -O3 -march=native -DNDEBUG)

# Set release mode
set(CMAKE_BUILD_TYPE Release)
