#pragma once

#include <cuda_runtime.h>
#include <stdlib.h>

// In the baseline kernel, the total number of threads is the same as array
// length. The partial sum computed at each block is collected at the 0th
// position of block, and then written to d_out.
__global__ void baseline_reduce_kernel(float *d_in, float *d_out, const int N) {

  const int tid = threadIdx.x;
  float *data = d_in + blockIdx.x * blockDim.x;

  // At each loop, the threads with tid < offset accumulates data[tid] +
  // data[tid + offset].
  for (int offset = (blockDim.x >> 1); offset > 0; offset >>= 1) {
    if (tid < offset) {
      data[tid] += data[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    d_out[blockIdx.x] = data[0];
  }
}