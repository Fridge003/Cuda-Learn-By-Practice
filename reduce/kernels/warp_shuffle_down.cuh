#pragma once

#include <cuda_runtime.h>
#include <stdlib.h>

#define WARPSIZE 32

__device__ __forceinline__ float warpReduceSum(float sum, int thread_num) {
  if (thread_num >= 32)
    sum += __shfl_down_sync(0xffffffff, sum, 16);
  if (thread_num >= 16)
    sum += __shfl_down_sync(0xffffffff, sum, 8);
  if (thread_num >= 8)
    sum += __shfl_down_sync(0xffffffff, sum, 4);
  if (thread_num >= 4)
    sum += __shfl_down_sync(0xffffffff, sum, 2);
  if (thread_num >= 2)
    sum += __shfl_down_sync(0xffffffff, sum, 1);
  return sum;
}

// Based on warp_unrolling kernel, this kernel uses the __shfl_down_sync
// primitive to accelerate the reduction operation inside the same warp. This
// kernel is close to the BlockReduceSum kernel implemented in Pytorch:
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/block_reduce.cuh
template <const int BLOCKSIZE, const int NUM_PER_THREAD>
__global__ void warp_shuffle_down_reduce_kernel(float *d_in, float *d_out,
                                                const int N) {

  const int tid = threadIdx.x;
  const int idx = blockIdx.x * (blockDim.x * NUM_PER_THREAD) + threadIdx.x;

  float partial_sum = 0.0;
#pragma unroll
  for (int iter = 0; iter < NUM_PER_THREAD; ++iter) {
    partial_sum +=
        (idx + iter * BLOCKSIZE < N) ? d_in[idx + iter * BLOCKSIZE] : 0.0;
  }

  __shared__ float warp_sum[WARPSIZE];
  const int laneIdx = tid % WARPSIZE;
  const int warpIdx = tid / WARPSIZE;

  // First, each warp does reduction and stores the result at warp_sum[warpIdx].
  float tmp_sum = warp_reduce(partial_sum, WARPSIZE);
  if (laneIdx == 0) {
    warp_sum[warpIdx] = tmp_sum;
  }
  __syncthreads();

  // Then, reducing the warp level sums using the first warp.
  tmp_sum = (tid * WARPSIZE < BLOCKSIZE) ? warp_sum[tid] : 0.0;
  if (warpIdx == 0) {
    tmp_sum = warp_reduce(tmp_sum, (BLOCKSIZE / WARPSIZE));
  }

  // Finally, the thread with 0 tid write back result to global memory.
  if (tid == 0) {
    d_out[blockIdx.x] = tmp_sum;
  }
}