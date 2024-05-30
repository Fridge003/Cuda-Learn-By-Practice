#pragma once

#include <cuda_runtime.h>
#include <stdlib.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// Current kernel assumes that M, N, K are multiples of BlockSize.
// General case of M, N, K is to be implemented.
template <const int BLOCKSIZE>
__global__ void shared_memory_cache_blocking_gemm_kernel(
    float *__restrict__ a, float *__restrict__ b, float *__restrict__ c,
    const int M, const int N, const int K) {

  // Current block is responsible for the calculation of submatrix
  // c[block_row_offset: block_row_offset + BLOCKSIZE, block_col_offset:
  // block_col_offset + BLOCKSIZE].
  const int block_row_offset = blockIdx.x * BLOCKSIZE;
  const int block_col_offset = blockIdx.y * BLOCKSIZE;

  // Each thread computes one element of c.
  // Current thread computes c[block_row_offset + thread_row, block_col_offset +
  // thread_col].
  int thread_row = threadIdx.x / BLOCKSIZE;
  int thread_col = threadIdx.x % BLOCKSIZE;

  // Advance pointer A, B, C to the starter position of submatrix.
  // So the position of block can be transparent.
  float *A = a, *B = b, *C = c;
  A += OFFSET(block_row_offset, 0, K);
  B += OFFSET(0, block_col_offset, N);
  C += OFFSET(block_row_offset, block_col_offset, N);

  // Allocate shared memory.
  __shared__ float A_s[BLOCKSIZE * BLOCKSIZE];
  __shared__ float B_s[BLOCKSIZE * BLOCKSIZE];

  float thread_sum = 0.0;
  // The outer loop goes through rows of A and columns of B.
  for (int block_idx = 0; block_idx < K; block_idx += BLOCKSIZE) {

    // Fetching data needed in current step from A, B to shared memory A_s, B_s.
    A_s[OFFSET(thread_row, thread_col, BLOCKSIZE)] =
        A[OFFSET(thread_row, thread_col, K)];
    B_s[OFFSET(thread_row, thread_col, BLOCKSIZE)] =
        B[OFFSET(thread_row, thread_col, N)];
    __syncthreads();

    // In the inner loop each thread updates the dot product it maintains.
    for (int dot_idx = 0; dot_idx < BLOCKSIZE; ++dot_idx) {
      thread_sum += A_s[OFFSET(thread_row, dot_idx, BLOCKSIZE)] *
                    B_s[OFFSET(dot_idx, thread_col, BLOCKSIZE)];
    }

    // Needs to sync again to avoid faster threads updating A_s, B_s before
    // slower threads finish updating sum.
    __syncthreads();

    // Advance A, B to the next position.
    A += BLOCKSIZE;
    B += BLOCKSIZE * N;
  }

  // Write result to C
  C[OFFSET(thread_row, thread_col, N)] = thread_sum;
}