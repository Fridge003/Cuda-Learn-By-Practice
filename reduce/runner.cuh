#pragma once

#include <cassert>
#include <cublas_v2.h>
#include <math.h>
#include <stdio.h>
#include <string>

#include "kernels.cuh"

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

double run_cpu_reduce(float *arr, int n) {
  double sum = 0.0;
  for (int i = 0; i < n; ++i) {
    sum += arr[i];
  }
  return sum;
}

void run_baseline_reduce(float *d_in, float *d_out, int n, int block_size) {
  int grid_size = CEIL_DIV(n, block_size);
  naive_gemm_kernel<<<grid_size, block_size>>>(d_in, d_out, n);
}

bool run_kernel(float *d_in, float *d_out, int n, int block_size,
                const std::string &kernel) {

  bool valid_kernel = false;

  if (kernel == "baseline") {
    run_naive_reduce(d_in, d_out, n, block_size);
    valid_kernel = true;
  }

  if (!valid_kernel) {
    printf("Invalid kernel type: %s\n", kernel.c_str());
  }

  return valid_kernel;
}