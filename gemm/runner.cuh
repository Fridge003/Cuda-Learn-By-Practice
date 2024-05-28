#pragma once

#include <cassert>
#include <cublas_v2.h>
#include <math.h>
#include <stdio.h>
#include <string>

#include "kernels.cuh"

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

void run_cublas_gemm(float *A, float *B, float *C, int m, int n, int k) {
  cublasHandle_t handle;
  cublasCreate(&handle);
  float alpha = 1.0f;
  float beta = 0.0f;
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, n, A, k,
              &beta, C, n);
  cublasDestroy(handle);
}

void run_naive_gemm(float *A, float *B, float *C, int m, int n, int k) {
  dim3 grid_size(CEIL_DIV(m, 32), CEIL_DIV(n, 32));
  dim3 block_size(32, 32);
  naive_gemm_kernel<<<grid_size, block_size>>>(A, B, C, m, n, k);
}

void run_global_memory_coalescing_kernel(float *A, float *B, float *C, int m,
                                         int n, int k) {
  const int BLOCKSIZE = 32;
  dim3 grid_size(CEIL_DIV(m, BLOCKSIZE), CEIL_DIV(n, BLOCKSIZE));
  dim3 block_size(BLOCKSIZE * BLOCKSIZE);
  global_memory_coalescing_gemm_kernel<BLOCKSIZE>
      <<<grid_size, block_size>>>(A, B, C, m, n, k);
}

void run_shared_memory_cache_blocking_kernel(float *A, float *B, float *C,
                                             int m, int n, int k) {
  const int BLOCKSIZE = 32;
  dim3 grid_size(CEIL_DIV(m, BLOCKSIZE), CEIL_DIV(n, BLOCKSIZE));
  dim3 block_size(BLOCKSIZE * BLOCKSIZE);
  shared_memory_cache_blocking_gemm_kernel<BLOCKSIZE>
      <<<grid_size, block_size>>>(A, B, C, m, n, k);
}

void run_one_d_block_tiling_kernel(float *A, float *B, float *C, int m, int n,
                                   int k) {

  // Sometimes tuning the BM/BN/BK/TM here might cause boost of performance, it
  // depends on device settings.
  const int BM = 64;
  const int BN = 64;
  const int BK = 8;
  const int TM = 8;

  // Here to make the number of threads per block equal to the number of
  // elements in shared memory, TM * BK == BM == BN should hold.
  assert(TM * BK == BM);
  assert(TM * BK == BN);

  dim3 grid_size(CEIL_DIV(m, BM), CEIL_DIV(n, BN));
  dim3 block_size(BM * BN / TM);
  one_d_block_tiling_gemm_kernel<BM, BN, BK, TM>
      <<<grid_size, block_size>>>(A, B, C, m, n, k);
}

void run_two_d_block_tiling_kernel(float *A, float *B, float *C, int m, int n,
                                   int k) {

  // Sometimes tuning the BM/BN/BK/TM/TN here might cause boost of performance,
  // it depends on device settings.
  const int BM = 128;
  const int BN = 128;
  const int BK = 8;
  const int TM = 8;
  const int TN = 8;

  dim3 grid_size(CEIL_DIV(m, BM), CEIL_DIV(n, BN));
  dim3 block_size((BM * BN) / (TM * TN));
  two_d_block_tiling_gemm_kernel<BM, BN, BK, TM, TN>
      <<<grid_size, block_size>>>(A, B, C, m, n, k);
}

void run_vectorized_two_d_block_tiling_kernel(float *A, float *B, float *C,
                                              int m, int n, int k) {

  const int BM = 128;
  const int BN = 128;
  const int BK = 8;
  const int TM = 8;
  const int TN = 8;

  // Make assertions that make vectorization work.
  assert(BK % 4 == 0);
  assert(BN % 4 == 0);
  assert(TN % 4 == 0);

  dim3 grid_size(CEIL_DIV(m, BM), CEIL_DIV(n, BN));
  dim3 block_size((BM * BN) / (TM * TN));
  vectorized_two_d_block_tiling_gemm_kernel<BM, BN, BK, TM, TN>
      <<<grid_size, block_size>>>(A, B, C, m, n, k);
}

void run_double_buffering_kernel(float *A, float *B, float *C, int m, int n,
                                 int k) {

  const int BM = 128;
  const int BN = 128;
  const int BK = 8;
  const int TM = 8;
  const int TN = 8;

  dim3 grid_size(CEIL_DIV(m, BM), CEIL_DIV(n, BN));
  dim3 block_size((BM * BN) / (TM * TN));
  double_buffering_gemm_kernel<BM, BN, BK, TM, TN>
      <<<grid_size, block_size>>>(A, B, C, m, n, k);
}

void run_bank_conflict_avoiding_kernel(float *A, float *B, float *C, int m,
                                       int n, int k) {

  const int BM = 128;
  const int BN = 128;
  const int BK = 8;
  const int TM = 8;
  const int TN = 8;

  dim3 grid_size(CEIL_DIV(m, BM), CEIL_DIV(n, BN));
  dim3 block_size((BM * BN) / (TM * TN));
  bank_conflict_avoiding_gemm_kernel<BM, BN, BK, TM, TN>
      <<<grid_size, block_size>>>(A, B, C, m, n, k);
}

bool run_kernel(float *A, float *B, float *C, int m, int n, int k,
                const std::string &kernel) {

  bool valid_kernel = false;

  if (kernel == "cublas") {
    run_cublas_gemm(A, B, C, m, n, k);
    valid_kernel = true;
  }

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

  if (kernel == "1D_block_tiling") {
    run_one_d_block_tiling_kernel(A, B, C, m, n, k);
    valid_kernel = true;
  }

  if (kernel == "2D_block_tiling") {
    run_two_d_block_tiling_kernel(A, B, C, m, n, k);
    valid_kernel = true;
  }

  if (kernel == "vectorized_2D_block_tiling") {
    run_vectorized_two_d_block_tiling_kernel(A, B, C, m, n, k);
    valid_kernel = true;
  }

  if (kernel == "double_buffering") {
    run_double_buffering_kernel(A, B, C, m, n, k);
    valid_kernel = true;
  }

  if (kernel == "bank_conflict_avoiding") {
    run_bank_conflict_avoiding_kernel(A, B, C, m, n, k);
    valid_kernel = true;
  }

  if (!valid_kernel) {
    printf("Invalid kernel type: %s\n", kernel.c_str());
  }

  return valid_kernel;
}