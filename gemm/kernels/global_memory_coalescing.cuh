#pragma once

#include <stdlib.h>
#include <cuda_runtime.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// Making threads in the same warp visit continual memory of matrix A, by transposing the role of m and n in naive kernel.
__global__ void global_memory_coalescing_gemm_kernel(float* __restrict__ a, float* __restrict__ b, float* __restrict__ c, 
                                                     const int M, const int N, const int K) {

  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (m < M && n < N) {
    float sum = 0.0;
    for (int k = 0; k < K; ++k) {
        sum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
    }
    c[OFFSET(m, n, N)] = sum;
  }
}