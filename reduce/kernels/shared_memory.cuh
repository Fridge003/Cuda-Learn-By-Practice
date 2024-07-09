#pragma once

#include <cuda_runtime.h>
#include <stdlib.h>

// Based on the baseline kernel, current kernel will first fetch the elements
// to shared memory before doing any reduction. In this way, the latency of
// accessing memory will be greatly decreased.
template <const int BLOCKSIZE>
__global__ void shared_memory_reduce_kernel(float *d_in, float *d_out,
                                            const int N) {

  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Each thread carries one element from global memory to shared memory.
  __shared__ float d_s[BLOCKSIZE];
  d_s[tid] = (idx < N) ? d_in[idx] : 0.0;
  __syncthreads();

  for (int offset = (blockDim.x >> 1); offset > 0; offset >>= 1) {
    if (tid < offset) {
      d_s[tid] += d_s[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    d_out[blockIdx.x] = d_s[0];
  }
}