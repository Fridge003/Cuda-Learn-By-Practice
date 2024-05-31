#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

const int warp_size = 32;

// This kernel implements warp-tiling on the basis of double_buffering
// kernel. Warp tiling technique makes 32 threads in the same warp compute
// a submatrix of c, thus enhancing locality and avoiding bank conflicts.

// The detailed idea is explained in Nvidia Cutlass Blog:
// https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/

// BM, BN, BK: Size of tiling at block level. Size of SMEM: 2 (double
// buffering) * BK * BM/BN. WM_OUT, WN_OUT: Size of tiling at warp level.
// Each warp computes an area with WM_OUT * WN_OUT elements. WM_IN, WN_IN:
// During computation, every warp will go through a double loop to fully
//               reuse the contents of registers reg_M/reg_N; Inside each
//               loop, a WM_IN * WN_IN area is covered.
// TM, TN: Size of tiling at thread level. Each thread computes TM * TN
// area.
template <const int BM, const int BN, const int BK, const int WM_OUT,
          const int WN_OUT, const int WM_IN, const int WN_IN, const int TM,
          const int TN>
__global__ void warp_tiling_gemm_kernel(float *__restrict__ a,
                                        float *__restrict__ b,
                                        float *__restrict__ c, const int M,
                                        const int N, const int K) {

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

  // Current warp is responsible for the calculation of submatrix:
  // C[warp_row * WM_OUT: (warp_row + 1) * WM_OUT,
  //   warp_col * WN_OUT: (warp_col + 1) * WN_OUT].
  const int current_warp_idx = threadIdx.x / warp_size;
  const int num_warp_col = BN / WN_OUT;
  const int warp_col = current_warp_idx % num_warp_col;
  const int warp_row = current_warp_idx / num_warp_col;

  // Number of inner iterations in every warp.
  constexpr int num_warp_iter_row = WM_OUT / WM_IN;
  constexpr int num_warp_iter_col = WN_OUT / WN_IN;

  // Postion of each thread inside warp.
  const int num_thread_col_in_warp = WN_IN / TN;
  const int current_thread_idx_in_warp = threadIdx.x % warp_size;
  const int thread_col_in_warp =
      current_thread_idx_in_warp % num_thread_col_in_warp;
  const int thread_row_in_warp =
      current_thread_idx_in_warp / num_thread_col_in_warp;

  // Allocate shared memory.
  __shared__ float A_s[2][BK * BM];
  __shared__ float B_s[2][BK * BN];

  // The scheme of loading from GMEM to SMEM doesn't need to change.
  const int num_thread_block = blockDim.x;

  const int load_col_A = threadIdx.x % (BK / 4);
  const int load_row_A_start = threadIdx.x / (BK / 4);
  const int num_load_A_per_thread = BM * BK / num_thread_block / 4;
  const int load_row_A_stride = BM / num_load_A_per_thread;

  const int load_col_B = threadIdx.x % (BN / 4);
  const int load_row_B_start = threadIdx.x / (BN / 4);
  const int num_load_B_per_thread = BN * BK / num_thread_block / 4;
  const int load_row_B_stride = BK / num_load_B_per_thread;

  // Allocate registers for the results managed by current thread.
  // Here each thread computes several TM * TN tiles, so the results are stored
  // as 2D array.
  float thread_results[num_warp_iter_row * TM * num_warp_iter_col * TN] = {0.0};

  // The registers reg_M and reg_N should be expanded by the number of
  // iterations inside each warp tile.
  float reg_M[2][num_warp_iter_row * TM] = {0.0};
  float reg_N[2][num_warp_iter_col * TN] = {0.0};

  int buffer_idx = 0;
  for (int block_idx = 0; block_idx < K; block_idx += BK) {

    // Fetching data needed in current step from A, B to shared memory A_s, B_s.
    // The loading process is the same as double_buffering kernel.
    for (int load_row_A = load_row_A_start; load_row_A < BM;
         load_row_A += load_row_A_stride) {
      float4 loaded_bytes =
          FETCH_FLOAT4(A[OFFSET(load_row_A, load_col_A * 4, K)]);

      // Load data into current buffer of shared memory.
      A_s[buffer_idx][OFFSET(load_col_A * 4 + 0, load_row_A, BM)] =
          loaded_bytes.x;
      A_s[buffer_idx][OFFSET(load_col_A * 4 + 1, load_row_A, BM)] =
          loaded_bytes.y;
      A_s[buffer_idx][OFFSET(load_col_A * 4 + 2, load_row_A, BM)] =
          loaded_bytes.z;
      A_s[buffer_idx][OFFSET(load_col_A * 4 + 3, load_row_A, BM)] =
          loaded_bytes.w;
    }

    for (int load_row_B = load_row_B_start; load_row_B < BK;
         load_row_B += load_row_B_stride) {
      FETCH_FLOAT4(B_s[buffer_idx][OFFSET(load_row_B, load_col_B * 4, BN)]) =
          FETCH_FLOAT4(B[OFFSET(load_row_B, load_col_B * 4, N)]);
    }
    __syncthreads();

    for (int dot_idx = 0; dot_idx < BK; ++dot_idx) {

      // Prefilling the registers with all needed elements.
      // Use float4 to vectorize loading.
      for (int inner_iter_row = 0; inner_iter_row < num_warp_iter_row;
           ++inner_iter_row) {
        for (int reg_idx = 0; reg_idx < TM; reg_idx += 4) {
          int row_offset = warp_row * WM_OUT + inner_iter_row * WM_IN +
                           thread_row_in_warp * TM + reg_idx;
          FETCH_FLOAT4(reg_M[buffer_idx][inner_iter_row * TM + reg_idx]) =
              FETCH_FLOAT4(A_s[buffer_idx][OFFSET(dot_idx, row_offset, BM)]);
        }
      }

      for (int inner_iter_col = 0; inner_iter_col < num_warp_iter_col;
           ++inner_iter_col) {
        for (int reg_idx = 0; reg_idx < TN; reg_idx += 4) {
          int col_offset = warp_col * WN_OUT + inner_iter_col * WN_IN +
                           thread_col_in_warp * TN + reg_idx;
          FETCH_FLOAT4(reg_N[buffer_idx][inner_iter_col * TN + reg_idx]) =
              FETCH_FLOAT4(B_s[buffer_idx][OFFSET(dot_idx, col_offset, BN)]);
        }
      }

      // Calculate outer products using elements stored in reg_M and reg_N,
      // and add them to thread_results.
      // There are totally num_warp_iter_row * num_warp_iter_col(two outer
      // loops) sets of outer products to be computed.
      for (int inner_iter_row = 0; inner_iter_row < num_warp_iter_row;
           ++inner_iter_row) {
        for (int inner_iter_col = 0; inner_iter_col < num_warp_iter_col;
             ++inner_iter_col) {
          for (int res_idx_M = 0; res_idx_M < TM; ++res_idx_M) {
            for (int res_idx_N = 0; res_idx_N < TN; ++res_idx_N) {
              int row_offset = inner_iter_row * TM + res_idx_M;
              int col_offset = inner_iter_col * TN + res_idx_N;
              thread_results[OFFSET(row_offset, col_offset,
                                    (num_warp_iter_col * TN))] +=
                  reg_M[buffer_idx][row_offset] * reg_N[buffer_idx][col_offset];
            }
          }
        }
      }
    }

    // Advance A, B to the next position.
    A += BK;
    B += BK * N;

    // Update index of buffer.
    buffer_idx = 1 - buffer_idx;
  }

  // Write results back to C.
  for (int inner_iter_row = 0; inner_iter_row < num_warp_iter_row;
       ++inner_iter_row) {
    for (int inner_iter_col = 0; inner_iter_col < num_warp_iter_col;
         ++inner_iter_col) {
      for (int res_idx_M = 0; res_idx_M < TM; ++res_idx_M) {
        for (int res_idx_N = 0; res_idx_N < TN; res_idx_N += 4) {
          int reg_row_offset = inner_iter_row * TM + res_idx_M;
          int reg_col_offset = inner_iter_col * TN + res_idx_N;
          int gmem_row_offset = warp_row * WM_OUT + inner_iter_row * WM_IN +
                                thread_row_in_warp * TM + res_idx_M;
          int gmem_col_offset = warp_col * WN_OUT + inner_iter_col * WN_IN +
                                thread_col_in_warp * TN + res_idx_N;
          FETCH_FLOAT4(C[OFFSET(gmem_row_offset, gmem_col_offset, N)]) =
              FETCH_FLOAT4(thread_results[OFFSET(reg_row_offset, reg_col_offset,
                                                 (num_warp_iter_col * TN))]);
        }
      }
    }
  }
}
