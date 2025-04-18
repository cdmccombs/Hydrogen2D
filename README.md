To run on perlmutter:
```
module load nvidia/24.5
module load cudatoolkit/12.4
module load kokkos-gpu/4.3.00
module load cmake/3.30.2
```

```
# Set environment variables for the correct compiler
export CC=cc
export CXX=CC
```
```
cmake .. -D Kokkos_ENABLE_CUDA=ON
make -j 4
```
