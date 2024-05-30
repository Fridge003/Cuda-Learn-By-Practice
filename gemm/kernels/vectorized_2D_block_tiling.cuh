#pragma once

#include <cuda_runtime.h>
#include <stdlib.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

// The main logic of this kernel is the same as 2D_block_tiling kernel,
// except that the loading process are vectorized for acceleration purpose.
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void vectorized_two_d_block_tiling_gemm_kernel(
    float *__restrict__ a, float *__restrict__ b, float *__restrict__ c,
    const int M, const int N, const int K) {

  // Current block is responsible for the calculation of submatrix
  // c[block_row_offset: block_row_offset + BM, block_col_offset:
  // block_col_offset + BN].
  const int block_row_offset = blockIdx.x * BM;
  const int block_col_offset = blockIdx.y * BN;

  // Advance pointer A, B, C to the starter position of submatrix.
  // So the position of block can be transparent.
  float *A = a, *B = b, *C = c;
  A += OFFSET(block_row_offset, 0, K);
  B += OFFSET(0, block_col_offset, N);
  C += OFFSET(block_row_offset, block_col_offset, N);

  // Number of threads in each row/col of block.
  const int num_thread_row = BM / TM;
  const int num_thread_col = BN / TN;
  const int num_thread_block = num_thread_row * num_thread_col;

  // Current thread computes an 2D submatrix
  // c[block_row_offset + thread_row * TM: block_row_offset + (thread_row + 1) *
  // TM,
  //   block_col_offset + thread_col * TN: block_col_offset + (thread_col + 1) *
  //   TN].
  const int thread_row = threadIdx.x / num_thread_col;
  const int thread_col = threadIdx.x % num_thread_col;

  // Allocate shared memory. Here A_s is transposed to enable 128-bit load
  // LDS.128 after compilation.
  __shared__ float A_s[BK * BM];
  __shared__ float B_s[BK * BN];

  // When loading elements of A from GMEM to SMEM, each thread loads a certain
  // number of float4 variables with load_col_A as its column index to achieve
  // better GMEM coalescing. The number of float4 variables loaded by each
  // thread is num_load_A_per_thread.
  const int load_col_A = threadIdx.x % (BK / 4);
  const int load_row_A_start = threadIdx.x / (BK / 4);
  const int num_load_A_per_thread = BM * BK / num_thread_block / 4;
  const int load_row_A_stride = BM / num_load_A_per_thread;

  // The method of loading from B is the same as loading from A.
  const int load_col_B = threadIdx.x % (BN / 4);
  const int load_row_B_start = threadIdx.x / (BN / 4);
  const int num_load_B_per_thread = BN * BK / num_thread_block / 4;
  const int load_row_B_stride = BK / num_load_B_per_thread;

  // Allocate registers for the results managed by current thread.
  float thread_results[TM * TN] = {0.0};

  // Allocate registers for caches.
  float reg_M[TM];
  float reg_N[TN];

  // The outer loop goes through rows of A and columns of B.
  for (int block_idx = 0; block_idx < K; block_idx += BK) {

    // Fetching data needed in current step from A, B to shared memory A_s, B_s.
    // Each time a float4 variable is loaded from GMEM to SMEM, so 128-bit
    // loading LDG.E.128 is triggered during compilation.
    for (int load_row_A = load_row_A_start; load_row_A < BM;
         load_row_A += load_row_A_stride) {
      float4 loaded_bytes =
          FETCH_FLOAT4(A[OFFSET(load_row_A, load_col_A * 4, K)]);

      // Here A_s is transposed, so loaded column index and loaded row index
      // are shifted during saving to A_s.
      A_s[OFFSET(load_col_A * 4 + 0, load_row_A, BM)] = loaded_bytes.x;
      A_s[OFFSET(load_col_A * 4 + 1, load_row_A, BM)] = loaded_bytes.y;
      A_s[OFFSET(load_col_A * 4 + 2, load_row_A, BM)] = loaded_bytes.z;
      A_s[OFFSET(load_col_A * 4 + 3, load_row_A, BM)] = loaded_bytes.w;
    }

    for (int load_row_B = load_row_B_start; load_row_B < BK;
         load_row_B += load_row_B_stride) {
      FETCH_FLOAT4(B_s[OFFSET(load_row_B, load_col_B * 4, BN)]) =
          FETCH_FLOAT4(B[OFFSET(load_row_B, load_col_B * 4, N)]);
    }
    __syncthreads();

    // Each thread calculate results.
    // The dot_idx loop goes along axis of K.
    for (int dot_idx = 0; dot_idx < BK; ++dot_idx) {

      // Load inputs of outer product into tmp registers.
      for (int reg_idx = 0; reg_idx < TM; ++reg_idx) {
        reg_M[reg_idx] = A_s[OFFSET(dot_idx, (thread_row * TM + reg_idx), BM)];
      }

      for (int reg_idx = 0; reg_idx < TN; ++reg_idx) {
        reg_N[reg_idx] = B_s[OFFSET(dot_idx, (thread_col * TN + reg_idx), BN)];
      }

      // Calculate outer product of reg_M and reg_N, and add it to
      // thread_results.
      for (int res_idx_M = 0; res_idx_M < TM; ++res_idx_M) {
        for (int res_idx_N = 0; res_idx_N < TN; ++res_idx_N) {
          thread_results[OFFSET(res_idx_M, res_idx_N, TN)] +=
              reg_M[res_idx_M] * reg_N[res_idx_N];
        }
      }
    }
    __syncthreads();

    // Advance A, B to the next position.
    A += BK;
    B += BK * N;
  }

  // Write results to C.
  // The process of writing back to GMEM is also vectorized.
  for (int res_idx_M = 0; res_idx_M < TM; ++res_idx_M) {
    for (int res_idx_N = 0; res_idx_N < TN; res_idx_N += 4) {
      FETCH_FLOAT4(C[OFFSET((thread_row * TM + res_idx_M),
                            (thread_col * TN + res_idx_N), N)]) =
          FETCH_FLOAT4(thread_results[OFFSET(res_idx_M, res_idx_N, TN)]);
    }
  }
}
