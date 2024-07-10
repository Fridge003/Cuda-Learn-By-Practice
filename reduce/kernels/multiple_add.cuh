#pragma once

#include <cuda_runtime.h>
#include <stdlib.h>

// Based on the shared memory kernel, this kernel makes each thread
// first compute the sum of NUM_PER_THREAD elements before writing it to shared
// memory. This increases the usage rate of each thread, thus avoiding idle
// threads.
template <const int BLOCKSIZE, const int NUM_PER_THREAD>
__global__ void multiple_add_reduce_kernel(float *d_in, float *d_out,
                                           const int N) {
  __shared__ float d_s[BLOCKSIZE];

  const int tid = threadIdx.x;
  const int idx = blockIdx.x * (blockDim.x * NUM_PER_THREAD) + threadIdx.x;

  // Accumulate the sum of NUM_PER_THREAD elements before carrying it to d_s.
  float partial_sum = 0.0;
  for (int iter = 0; iter < NUM_PER_THREAD; ++iter) {
    partial_sum +=
        (idx + iter * BLOCKSIZE < N) ? d_in[idx + iter * BLOCKSIZE] : 0.0;
  }
  d_s[tid] = partial_sum;
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