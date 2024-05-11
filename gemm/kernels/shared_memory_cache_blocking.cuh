#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// Current kernel assumes that M, N, K are multiples of BlockSize.
// General case of M, N, K is to be implemented.
template <const int BLOCKSIZE>
__global__ void shared_memory_cache_blocking_gemm_kernel(
    float *__restrict__ a, float *__restrict__ b, float *__restrict__ c,
    const int M, const int N, const int K) {

  // Current block is responsible for the calculation of submatrix
  // c[row_start:row_start + BLOCKSIZE, col_start:col_start + BLOCKSIZE].
  const int row_start = blockIdx.x * BLOCKSIZE;
  const int col_start = blockIdx.y * BLOCKSIZE;

  // Each thread computes one element of c.
  // Current thread computes c[row_start + thread_row, col_start + thread_col].
  int thread_row = threadIdx.x / BLOCKSIZE;
  int thread_col = threadIdx.x % BLOCKSIZE;

  // Advance pointer A, B, C to the starter position of submatrix.
  float *A = a, *B = b, *C = c;
  A += OFFSET(row_start, 0, K);
  B += OFFSET(0, col_start, N);
  C += OFFSET(row_start, col_start, N);

  // Allocate shared memory.
  __shared__ float A_s[BLOCKSIZE * BLOCKSIZE];
  __shared__ float B_s[BLOCKSIZE * BLOCKSIZE];

  float tmp = 0.0;
  // The outer loop goes through rows of A and columns of B.
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {

    // Fetching data needed in current step from A, B to shared memory A_s, B_s.
    A_s[OFFSET(thread_row, thread_col, BLOCKSIZE)] =
        A[OFFSET(thread_row, thread_col, K)];
    B_s[OFFSET(thread_row, thread_col, BLOCKSIZE)] =
        B[OFFSET(thread_row, thread_col, N)];
    __syncthreads();

    // Advance A, B to the next position.
    A += BLOCKSIZE;
    B += BLOCKSIZE * N;

    // In the inner loop each thread updates the dot product it maintains.
    for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
      tmp += A_s[OFFSET(thread_row, dotIdx, BLOCKSIZE)] *
             B_s[OFFSET(dotIdx, thread_col, BLOCKSIZE)];
    }

    // Needs to sync again to avoid faster threads updating A_s, B_s before
    // slower threads finish updating sum.
    __syncthreads();
  }

  // Write result to C
  C[OFFSET(thread_row, thread_col, N)] = tmp;
}