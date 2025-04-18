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
