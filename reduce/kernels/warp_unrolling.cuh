#pragma once

#include <cuda_runtime.h>
#include <stdlib.h>

#define WARPSIZE 32

// The helper function for reducing threads in the same warp.
// Here the "volatile" keyword is necessary, since it prevents thread from
// using incorrect results cached in register.
__device__ void warp_reduce(volatile float *cache, int tid) {
  cache[tid] += cache[tid + 32];
  cache[tid] += cache[tid + 16];
  cache[tid] += cache[tid + 8];
  cache[tid] += cache[tid + 4];
  cache[tid] += cache[tid + 2];
  cache[tid] += cache[tid + 1];
}

// The instructions are executed synchronously at warp level.
// So the thread synchronization can be avoided when working threads are in the
// same warp.
template <const int BLOCKSIZE, const int NUM_PER_THREAD>
__global__ void multiple_add_reduce_kernel(float *d_in, float *d_out,
                                           const int N) {
  __shared__ float d_s[BLOCKSIZE];

  const int tid = threadIdx.x;
  const int idx = blockIdx.x * (blockDim.x * NUM_PER_THREAD) + threadIdx.x;

  float partial_sum = 0.0;
#pragma unroll
  for (int iter = 0; iter < NUM_PER_THREAD; ++iter) {
    partial_sum +=
        (idx + iter * BLOCKSIZE < N) ? d_in[idx + iter * BLOCKSIZE] : 0.0;
  }
  d_s[tid] = partial_sum;
  __syncthreads();

  // When working threads are not in the same warp, synchronization is needed.
  for (int offset = (blockDim.x >> 1); offset > WARPSIZE; offset >>= 1) {
    if (tid < offset) {
      d_s[tid] += d_s[tid + offset];
    }
    __syncthreads();
  }

  // When working threads are in the same warp, call warp_reduce function.
  if (tid < WARPSIZE)
    warp_reduce(d_s, tid);

  if (tid == 0) {
    d_out[blockIdx.x] = d_s[0];
  }
}