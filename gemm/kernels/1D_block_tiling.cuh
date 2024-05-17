#pragma once

#include <cuda_runtime.h>
#include <stdlib.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// The kernel is almost the same as shared_memory_cache_blocking kernel,
// except that each threads computes TM elements of c instead of one.
// Each block contains (BM * BM / TM) threads.

// Through lifting the number of elements per thread,
// the per-thread number of loading from shared memory can b divided by the size
// of tiling, and the arithemic intensity is increased.

template <const int BM, const int BN, const int BK, const int TM>
__global__ void
one_d_block_tiling_gemm_kernel(float *__restrict__ a, float *__restrict__ b,
                               float *__restrict__ c, const int M, const int N,
                               const int K) {

  // Current block is responsible for the calculation of submatrix
  // c[block_row_start: block_row_start + BM, block_col_start: block_col_start +
  // BN].
  const int block_row_start = blockIdx.x * BM;
  const int block_col_start = blockIdx.y * BN;

  // Current thread computes an 1D submatrix
  // c[block_row_start + thread_row * TM: block_row_start + (thread_row + 1) *
  // TM, block_col_start + thread_col].
  const int thread_row = threadIdx.x / BN;
  const int thread_col = threadIdx.x % BN;

  // Advance pointer A, B, C to the starter position of submatrix.
  float *A = a, *B = b, *C = c;
  A += OFFSET(block_row_start, 0, K);
  B += OFFSET(0, block_col_start, N);
  C += OFFSET(block_row_start, block_col_start, N);

  // The positions of elements in A and B to fetch from global memory to shared
  // memory.
  const int load_row_A =
      threadIdx.x / BK; // warp-level global memory coalescing
  const int load_col_A = threadIdx.x % BK;
  const int load_row_B = threadIdx.x / BN;
  const int load_col_B = threadIdx.x % BN;

  // Allocate shared memory.
  __shared__ float A_s[BM * BK];
  __shared__ float B_s[BK * BN];

  // Allocate registers for the results managed by current thread.
  float thread_results[TM] = {0.0};

  // The outer loop goes through rows of A and columns of B.
  for (int block_idx = 0; block_idx < K; block_idx += BK) {

    // Fetching data needed in current step from A, B to shared memory A_s, B_s.
    A_s[OFFSET(load_row_A, load_col_A, BK)] =
        A[OFFSET(load_row_A, load_col_A, K)];
    B_s[OFFSET(load_row_B, load_col_B, BN)] =
        B[OFFSET(load_row_B, load_col_B, N)];
    __syncthreads();

    // Each thread calculate results.
    // The dot_idx loop goes along columns of A and rows of B,
    // the res_idx loop iterates through each result to compute.
    for (int dot_idx = 0; dot_idx < BK; ++dot_idx) {

      // Load from B_s once and reuse for TM times.
      // This step is critical for the boost of performance.
      float B_tmp = B_s[OFFSET(dot_idx, thread_col, BN)];

      for (int res_idx = 0; res_idx < TM; ++res_idx) {
        thread_results[res_idx] +=
            A_s[OFFSET((thread_row * TM + res_idx), dot_idx, BK)] * B_tmp;
      }
    }
    __syncthreads();

    // Advance A, B to the next position.
    A += BK;
    B += BK * N;
  }

  // Write results to C
  for (int res_idx = 0; res_idx < TM; ++res_idx) {
    C[OFFSET((thread_row * TM + res_idx), thread_col, N)] =
        thread_results[res_idx];
  }
}
