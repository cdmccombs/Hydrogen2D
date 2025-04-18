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

# Create build directory if it doesn't exist
mkdir -p $BUILD_DIR
mkdir -p $RESULTS_DIR

# Load required modules
module purge
module load PrgEnv-gnu
module load cudatoolkit
module load kokkos-gpu/4.3.00
module load cmake

# Display loaded modules and compiler versions
echo "Loaded modules:"
module list

echo "Compiler versions:"
gcc --version
nvcc --version

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

# Check if build succeeded
if [ -f "hydrogen_simulator_2d" ]; then
    echo "Build successful! Running simulation..."
    srun -n 1 ./hydrogen_simulator_2d
    
    # Copy results to results directory
    cp *.dat $RESULTS_DIR/
    echo "Results saved to $RESULTS_DIR"
else
    echo "Build failed. Please check the error messages."
fi
