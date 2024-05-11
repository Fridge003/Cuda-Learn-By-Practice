#pragma once

#include <stdio.h>
#include <math.h>
#include <string>
#include <cublas_v2.h>

#include "kernels.cuh"

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

void run_cublas_gemm(float* A, float* B, float* C, int m, int n, int k) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, n, A, k, &beta, C, n);
    cublasDestroy(handle);
}

void run_naive_gemm(float* A, float* B, float* C, int m, int n, int k) {
    dim3 grid_size(CEIL_DIV(m, 32), CEIL_DIV(n, 32));   
    dim3 block_size(32, 32);
    naive_gemm_kernel<<<grid_size, block_size>>>(A, B, C, m, n, k);
}

void run_global_memory_coalescing_kernel(float* A, float* B, float* C, int m, int n, int k) {
    const int BLOCKSIZE = 32;
    dim3 grid_size(CEIL_DIV(m, BLOCKSIZE), CEIL_DIV(n, BLOCKSIZE));   
    dim3 block_size(BLOCKSIZE * BLOCKSIZE);
    global_memory_coalescing_gemm_kernel<BLOCKSIZE><<<grid_size, block_size>>>(A, B, C, m, n, k);
}

void run_shared_memory_cache_blocking_kernel(float* A, float* B, float* C, int m, int n, int k) {
    const int BLOCKSIZE = 32;
    dim3 grid_size(CEIL_DIV(m, BLOCKSIZE), CEIL_DIV(n, BLOCKSIZE));   
    dim3 block_size(BLOCKSIZE * BLOCKSIZE);
    shared_memory_cache_blocking_gemm_kernel<BLOCKSIZE><<<grid_size, block_size>>>(A, B, C, m, n, k);
}

bool run_kernel(float* A,
                float* B,
                float* C,
                int m,
                int n,
                int k,
                const std::string& kernel) {
    
    bool valid_kernel = false;

    if (kernel == "naive") {
        run_naive_gemm(A, B, C, m, n, k);
        valid_kernel = true;
    }

    if (kernel == "global_memory_coalescing") {
        run_global_memory_coalescing_kernel(A, B, C, m, n, k);
        valid_kernel = true;
    }

    if (kernel == "shared_memory_cache_blocking") {
        run_shared_memory_cache_blocking_kernel(A, B, C, m, n, k);
        valid_kernel = true;
    }

    if (kernel == "cublas") {
        run_cublas_gemm(A, B, C, m, n, k);
        valid_kernel = true;
    }

    if (!valid_kernel) {
        printf("Invalid kernel type: %s\n", kernel.c_str());
    }

    return valid_kernel;
}