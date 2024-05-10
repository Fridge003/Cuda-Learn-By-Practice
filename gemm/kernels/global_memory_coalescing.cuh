#pragma once

#include <stdlib.h>
#include <cuda_runtime.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// In this kernel, the block shape is defined as 1D (BLOCK_SIZE * BLOCK_SIZE), but regarded as 2D (BLOCK_SIZE, BLOCK_SIZE).
// In this way, the threads in the same warp can compute elements in the same row of output matrix.
template <const int BLOCKSIZE>
__global__ void global_memory_coalescing_gemm_kernel(float* __restrict__ a, float* __restrict__ b, float* __restrict__ c, 
                                                     const int M, const int N, const int K) {
  
  const uint m = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const uint n = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);                                                
  
  if (m < M && n < N) {
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
    }
    c[OFFSET(m, n, N)] = sum;
  }
}