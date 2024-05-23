#include <cassert>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "configs.h"
#include "runner.cuh"
#include "utils.cuh"

int main(int argc, char **argv) {

  std::vector<std::string> kernels_to_run;

  if (argc == 1) {
    kernels_to_run = registered_kernel;
  } else if (argc == 2) {
    // Get kernel number.
    int kernel_num = std::stoi(argv[1]);
    if (kernel_num < 0 || kernel_num >= registered_kernel.size()) {
      printf("Please enter a valid kernel number (0-%ld), valid kernels are as "
             "follows:\n",
             registered_kernel.size() - 1);
      for (int i = 0; i < registered_kernel.size(); ++i) {
        printf("Kernel %d: %s\n", i, registered_kernel[i].c_str());
      }
      exit(EXIT_FAILURE);
    }
    kernels_to_run.push_back(registered_kernel[kernel_num]);
  } else {
    printf("Too many arguments! Usage: ./gemm for testing of all kernels; "
           "./gemm [kernel idx] for testing one kernel.");
    exit(EXIT_FAILURE);
  }

  // Get device if set in environment variable.
  int deviceIdx = 0;
  if (getenv("DEVICE") != NULL) {
    deviceIdx = atoi(getenv("DEVICE"));
  }
  CUDA_CHECK(cudaSetDevice(deviceIdx));

  CudaDeviceInfo();

  for (int test_case = 0; test_case < mnk_list.size(); ++test_case) {
    print_border_line();

    const int m = mnk_list[test_case][0];
    const int n = mnk_list[test_case][1];
    const int k = mnk_list[test_case][2];

    printf("Test %d: m = %d, n = %d, k = %d\n", test_case, m, n, k);
    estimate_compute_and_IO_cost(m, n, k, device_fp32_compute_capacity_tflops,
                                 device_global_mem_bandwidth_GB_per_sec);

    // Allocate space for the matrices, the goal is to calculate A @ B and store
    // it in C h for host, d for device; h_C_ref is for correctness checking
    size_t size_A = m * k * sizeof(float);
    size_t size_B = k * n * sizeof(float);
    size_t size_C = m * n * sizeof(float);

    float *h_A, *h_B, *h_C, *h_C_ref, *d_A, *d_B, *d_C, *d_C_ref;
    h_A = (float *)malloc(size_A);
    h_B = (float *)malloc(size_B);
    h_C = (float *)malloc(size_C);
    h_C_ref = (float *)malloc(size_C);
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));
    CUDA_CHECK(cudaMalloc(&d_C_ref, size_C));

    // Initialize the matrices, and copy them to device.
    randomize_matrix(h_A, m * k);
    randomize_matrix(h_B, k * n);
    zero_init_matrix(h_C_ref, m * n);
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C_ref, h_C_ref, size_C, cudaMemcpyHostToDevice));

    // Store the correct result in d_C_ref, and copy back.
    run_cublas_gemm(d_A, d_B, d_C_ref, m, n, k);
    CUDA_CHECK(cudaMemcpy(h_C_ref, d_C_ref, size_C, cudaMemcpyDeviceToHost));

    // Test each kernel.
    for (const std::string &kernel : kernels_to_run) {
      printf("\nKernel: %s\n", kernel.c_str());

      zero_init_matrix(h_C, m * n);
      CUDA_CHECK(cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice));

      // Check kernel validity.
      bool valid_kernel = run_kernel(d_A, d_B, d_C, m, n, k, kernel);
      if (!valid_kernel)
        continue;

      // Check Correctness.
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
      bool correct = check_result_correctness(h_C, h_C_ref, m, n);
      if (!correct)
        continue;

      // Check Performance.
      check_performance(kernel, d_A, d_B, d_C, m, n, k, 5);
    }

    // Free Memory.
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
  }
  print_border_line();

  return 0;
}