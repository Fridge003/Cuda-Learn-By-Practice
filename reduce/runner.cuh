#pragma once

#include <cassert>
#include <cublas_v2.h>
#include <math.h>
#include <stdio.h>
#include <string>

#include "configs.h"
#include "kernels.cuh"

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

double run_cpu_reduce(float *arr, int n) {
  double sum = 0.0;
  for (int i = 0; i < n; ++i) {
    sum += arr[i];
  }
  return sum;
}

void run_baseline_reduce(float *d_in, float *d_out, int n) {
  int grid_size = CEIL_DIV(n, block_size);
  baseline_reduce_kernel<<<grid_size, block_size>>>(d_in, d_out, n);
}

void run_shared_memory_reduce(float *d_in, float *d_out, int n) {
  int grid_size = CEIL_DIV(n, block_size);
  shared_memory_reduce_kernel<block_size>
      <<<grid_size, block_size>>>(d_in, d_out, n);
}

void run_multiple_add_reduce(float *d_in, float *d_out, int n) {
  int grid_size = CEIL_DIV(n, (block_size * num_per_thread));
  multiple_add_reduce_kernel<block_size, num_per_thread>
      <<<grid_size, block_size>>>(d_in, d_out, n);
}

void run_warp_unrolling_reduce(float *d_in, float *d_out, int n) {
  const int warp_size = 32;
  assert(block_size >= warp_size);
  int grid_size = CEIL_DIV(n, (block_size * num_per_thread));
  warp_unrolling_reduce_kernel<block_size, num_per_thread>
      <<<grid_size, block_size>>>(d_in, d_out, n);
}

void run_warp_shuffle_down_reduce(float *d_in, float *d_out, int n) {
  const int warp_size = 32;
  assert((block_size >= warp_size) && (block_size <= (warp_size * warp_size)));
  int grid_size = CEIL_DIV(n, (block_size * num_per_thread));
  warp_shuffle_down_reduce_kernel<block_size, num_per_thread>
      <<<grid_size, block_size>>>(d_in, d_out, n);
}

bool run_kernel(float *d_in, float *d_out, int n, const std::string &kernel) {

  bool valid_kernel = false;

  if (kernel == "baseline") {
    run_baseline_reduce(d_in, d_out, n);
    valid_kernel = true;
  }

  if (kernel == "shared_memory") {
    run_shared_memory_reduce(d_in, d_out, n);
    valid_kernel = true;
  }

  if (kernel == "multiple_add") {
    run_multiple_add_reduce(d_in, d_out, n);
    valid_kernel = true;
  }

  if (kernel == "warp_unrolling") {
    run_warp_unrolling_reduce(d_in, d_out, n);
    valid_kernel = true;
  }

  if (kernel == "warp_shuffle_down") {
    run_warp_shuffle_down_reduce(d_in, d_out, n);
    valid_kernel = true;
  }

  if (!valid_kernel) {
    printf("Invalid kernel type: %s\n", kernel.c_str());
  }

  return valid_kernel;
}