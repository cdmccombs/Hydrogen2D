#!/bin/bash
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH -J hydrogen_sim
#SBATCH -A m558_g
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH --gpus=1
#SBATCH --gpu-bind=none
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o hydrogen.o%j
#SBATCH -e hydrogen.e%j

# Directory paths
BASE_DIR=/pscratch/sd/c/cmccombs/TDSE_sims/GPU_attempts
SRC_DIR=$BASE_DIR/hydrogen_kokkos
BUILD_DIR=$BASE_DIR/build/hydrogen_kokkos
RESULTS_DIR=$BASE_DIR/results

# Create directories if they don't exist
mkdir -p $SRC_DIR
mkdir -p $BUILD_DIR
mkdir -p $RESULTS_DIR

# Load required modules
module purge
module load PrgEnv-gnu
module load cudatoolkit
module load kokkos-gpu/4.3.00
module load cmake

echo "Loaded modules:"
module list

echo "Compiler versions:"
gcc --version
nvcc --version

# Create the CMakeLists.txt file
cat > $SRC_DIR/CMakeLists.txt << 'EOF'
cmake_minimum_required(VERSION 3.16)
project(HydrogenSimulator2D LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

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

# Set CUDA architecture for A100
set_property(TARGET hydrogen_simulator_2d PROPERTY CUDA_ARCHITECTURES 80)

# Disable relocatable device code to avoid conflict with Kokkos
set_property(TARGET hydrogen_simulator_2d PROPERTY CUDA_SEPARABLE_COMPILATION OFF)

# Add CUDA compiler flags
target_compile_options(hydrogen_simulator_2d PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-allow-unsupported-compiler>
)

# Link libraries
target_link_libraries(hydrogen_simulator_2d 
    Kokkos::kokkos
    CUDA::cudart
    CUDA::cufft
)

# Installation
install(TARGETS hydrogen_simulator_2d DESTINATION bin)
EOF

# Create the cuFFT wrapper
cat > $SRC_DIR/cufft_wrapper.cu << 'EOF'
#include <cufft.h>
#include <stdio.h>

extern "C" {

// Forward FFT implementation using cuFFT
void forward_fft_2d(void* data, int nx, int ny) {
    cufftHandle plan;
    cufftResult result = cufftPlan2d(&plan, nx, ny, CUFFT_Z2Z);
    if (result != CUFFT_SUCCESS) {
        printf("CUFFT Plan creation failed: %d\n", result);
        return;
    }
    
    result = cufftExecZ2Z(plan, (cufftDoubleComplex*)data, (cufftDoubleComplex*)data, CUFFT_FORWARD);
    if (result != CUFFT_SUCCESS) {
        printf("CUFFT forward transform failed: %d\n", result);
    }
    
    cufftDestroy(plan);
}

// Backward FFT implementation using cuFFT
void backward_fft_2d(void* data, int nx, int ny) {
    cufftHandle plan;
    cufftResult result = cufftPlan2d(&plan, nx, ny, CUFFT_Z2Z);
    if (result != CUFFT_SUCCESS) {
        printf("CUFFT Plan creation failed: %d\n", result);
        return;
    }
    
    result = cufftExecZ2Z(plan, (cufftDoubleComplex*)data, (cufftDoubleComplex*)data, CUFFT_INVERSE);
    if (result != CUFFT_SUCCESS) {
        printf("CUFFT inverse transform failed: %d\n", result);
    }
    
    cufftDestroy(plan);
}

}
EOF

# Create a simplified test program 
cat > $SRC_DIR/hydrogen_simulator_2d.cpp << 'EOF'
#include <iostream>
#include <Kokkos_Core.hpp>

// Forward declare cuFFT wrapper functions
extern "C" {
    void forward_fft_2d(void* data, int nx, int ny);
    void backward_fft_2d(void* data, int nx, int ny);
}

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        std::cout << "Successfully initialized Kokkos!" << std::endl;
        
        if (std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::Cuda>::value) {
            std::cout << "Kokkos is using CUDA for execution space!" << std::endl;
        } else {
            std::cout << "Warning: Kokkos is NOT using CUDA for execution space!" << std::endl;
        }
        
        // Create a small test
        const int N = 16;
        Kokkos::View<double**> d_data("data", N, N);
        
        // Initialize with a pattern
        Kokkos::parallel_for("init", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N, N}),
            KOKKOS_LAMBDA(const int i, const int j) {
                d_data(i, j) = i*N + j;
            }
        );
        
        std::cout << "Successfully created and initialized Kokkos View" << std::endl;
        
        // Test FFT functions if working
        std::cout << "Hydrogen simulator with Kokkos completed initial test!" << std::endl;
        std::cout << "You can now implement the full simulation code." << std::endl;
    }
    Kokkos::finalize();
    
    return 0;
}
EOF

# Navigate to build directory
cd $BUILD_DIR

# Clean any previous build artifacts
rm -rf *

# Configure with CMake
echo "Configuring with CMake..."
cmake -DCMAKE_BUILD_TYPE=Release \
      -DKokkos_ENABLE_CUDA=ON \
      $SRC_DIR

# Build
echo "Building the application..."
make -j 8

# Check if the build succeeded
if [ -f "hydrogen_simulator_2d" ]; then
    echo "Build successful! Running simulation..."
    
    # Run the test program
    srun -n 1 ./hydrogen_simulator_2d
    
    echo "Basic test complete. Ready to implement full simulation code."
else
    echo "Build failed. Please check the error messages."
fi