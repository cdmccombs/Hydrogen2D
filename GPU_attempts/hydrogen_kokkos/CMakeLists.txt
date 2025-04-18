cmake_minimum_required(VERSION 3.16)

# Set CUDA architecture globally BEFORE project()
set(CMAKE_CUDA_ARCHITECTURES 80) # <-- Add this line

project(HydrogenSimulator2D LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# CUDA Standard settings might not be needed if NVHPC handles it
# set(CMAKE_CUDA_STANDARD 14)
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Always use Release build for performance
set(CMAKE_BUILD_TYPE Release)

# Find packages
find_package(Kokkos REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Source files
set(SOURCES
    hydrogen_simulator_2d.cpp
    cufft_wrapper.cu
)

# Create executable
add_executable(hydrogen_simulator_2d ${SOURCES})

# Set CUDA architecture (Redundant now, but harmless)
# set_property(TARGET hydrogen_simulator_2d PROPERTY CUDA_ARCHITECTURES 80)

# Disable relocatable device code to avoid conflict with Kokkos
set_property(TARGET hydrogen_simulator_2d PROPERTY CUDA_SEPARABLE_COMPILATION OFF)

# Add CUDA compiler flags
target_compile_options(hydrogen_simulator_2d PRIVATE
    $<$<OR:$<COMPILE_LANGUAGE:CXX>,$<COMPILE_LANGUAGE:CUDA>>:-allow-unsupported-compiler>
)

# Note: This flag specifically applies to CUDA compilation steps. If the C++ compilation
# itself fails with a similar error (unlikely but possible), we might need to add it
# for CXX as well, but let's try just CUDA first.

# Link libraries
target_link_libraries(hydrogen_simulator_2d
    Kokkos::kokkos
    CUDA::cudart
    CUDA::cufft
)

# Installation
install(TARGETS hydrogen_simulator_2d DESTINATION bin)
