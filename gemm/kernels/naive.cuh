#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

void cpu_gemm(
    float* a, float* b, float* c, const int M, const int N, const int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0;
            for (int k = 0; k < K; k++) {
                sum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
            }
            c[OFFSET(m, n, N)] = sum;
        }
    }
}

__global__ void naive_gemm_kernel(float* __restrict__ a, float* __restrict__ b, float* __restrict__ c, 
                                  const int M, const int N, const int K) {

  int m = blockIdx.x * blockDim.x + threadIdx.x;
  int n = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (m < M && n < N) {
    float sum = 0.0;
    #pragma unroll
    for (int k = 0; k < K; ++k) {
        sum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
    }
    c[OFFSET(m, n, N)] = sum;
  }
}