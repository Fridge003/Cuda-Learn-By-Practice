#include <cassert>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "configs.h"
#include "runner.cuh"
#include "utils.cuh"

std::string kernel_idx_to_name(int kernel_idx) {
  if (kernel_idx < 0 || kernel_idx >= registered_kernel.size()) {
    printf("Please enter a valid kernel number (0-%ld), valid kernels are as "
           "follows:\n",
           registered_kernel.size() - 1);
    for (int i = 0; i < registered_kernel.size(); ++i) {
      printf("Kernel %d: %s\n", i, registered_kernel[i].c_str());
    }
    exit(EXIT_FAILURE);
  }
  return registered_kernel[kernel_idx];
}

void trigger_kernel_once(const std::string &kernel_to_run, const int n) {

  print_border_line();
  printf("Run kernel %s: n = %d, number of threads per block = %d\n",
         kernel_to_run.c_str(), n, block_size);

  // Allocate space for the array, h for host, d for device;
  // The length of d_out is the same as the number of blocks.
  size_t size_in = n * sizeof(float);
  size_t n_block = (n + block_size - 1) / block_size;
  size_t size_out = n_block * sizeof(float);

  float *h_in, *h_out, *d_in, *d_out;
  h_in = (float *)malloc(size_in);
  h_out = (float *)malloc(size_out);
  CUDA_CHECK(cudaMalloc(&d_in, size_in));
  CUDA_CHECK(cudaMalloc(&d_out, size_out));

  // Initialize the arrays and copy them to device.
  randomize_array(h_in, n);
  zero_init_array(h_out, n_block);
  CUDA_CHECK(cudaMemcpy(d_in, h_in, size_in, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_out, h_out, size_out, cudaMemcpyHostToDevice));

  // Run kernel and copy result from d_C back to h_C.
  bool valid_kernel = run_kernel(d_in, d_out, n, kernel_to_run);
  if (!valid_kernel)
    printf("Invalid kernel!\n");
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_out, d_out, size_out, cudaMemcpyDeviceToHost));

  // Free Memory.
  free(h_in);
  free(h_out);
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));

  print_border_line();
}

void run_tests(const std::vector<std::string> &kernels_to_run) {

  print_border_line();
  printf("GPU Bandwidth = %lf GB/s, number of threads per block = %d\n",
         device_global_mem_bandwidth_GB_per_sec, block_size);

  for (int test_case = 0; test_case < n_list.size(); ++test_case) {
    print_border_line();

    const int n = n_list[test_case];

    printf("Test %d: n = %d\n", test_case, n);

    // Allocate space for the array, h for host, d for device;
    // The length of d_out is the same as the number of blocks.
    size_t size_in = n * sizeof(float);
    size_t n_block = (n + block_size - 1) / block_size;
    size_t size_out = n_block * sizeof(float);

    float *h_in, *h_out, *d_in, *d_out;
    h_in = (float *)malloc(size_in);
    h_out = (float *)malloc(size_out);
    CUDA_CHECK(cudaMalloc(&d_in, size_in));
    CUDA_CHECK(cudaMalloc(&d_out, size_out));

    // Initialize the input array and store the correct result of reduction to
    // sum_ref.
    randomize_array(h_in, n);
    double sum_ref = run_cpu_reduce(h_in, n);

    // Test each kernel.
    for (const std::string &kernel : kernels_to_run) {
      printf("\nKernel: %s\n", kernel.c_str());

      // Clean the output array, and copy arrays to Cuda device.
      zero_init_array(h_out, n_block);
      CUDA_CHECK(cudaMemcpy(d_in, h_in, size_in, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_out, h_out, size_out, cudaMemcpyHostToDevice));

      // Check kernel validity.
      bool valid_kernel = run_kernel(d_in, d_out, n, kernel);
      if (!valid_kernel)
        continue;

      // Copy results back and check Correctness.
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemcpy(h_out, d_out, size_out, cudaMemcpyDeviceToHost));

      bool correct = check_result_correctness(h_out, sum_ref, n_block);
      if (!correct)
        continue;

      // Check Performance.
      check_performance(kernel, d_in, d_out, n, 10, 10);
    }

    // Free Memory.
    free(h_in);
    free(h_out);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
  }
  print_border_line();
}

int main(int argc, char **argv) {

  // Get device if set in environment variable.
  int deviceIdx = 0;
  if (getenv("DEVICE") != NULL) {
    deviceIdx = atoi(getenv("DEVICE"));
  }
  CUDA_CHECK(cudaSetDevice(deviceIdx));

  // Mode 0: list valid kernels.
  if ((argc >= 2) && (std::string(argv[1]) == "--list-kernels")) {
    for (int i = 0; i < registered_kernel.size(); ++i) {
      printf("Kernel %d: %s\n", i, registered_kernel[i].c_str());
    }
    return 0;
  }

  // Mode 1: Trigger the kernel once without any testing.
  if ((argc >= 2) && (std::string(argv[1]) == "--once")) {
    if (argc != 4) {
      printf("Too many or too few arguments! Usage: ./reduce --once "
             "[kernel_idx] [N]\n");
      exit(EXIT_FAILURE);
    } else {
      CudaDeviceInfo(); // Print device information.
      std::string kernel_to_run = kernel_idx_to_name(std::stoi(argv[2]));
      int n = std::stoi(argv[3]);
      trigger_kernel_once(kernel_to_run, n);
    }
    return 0;
  }

  // Mode 2: Do all the tests, including correctness check and performance
  // check.
  std::vector<std::string> collected_kernels;
  if (argc == 1) {
    collected_kernels = registered_kernel;
  } else if (argc == 2) {
    int kernel_idx = std::stoi(argv[1]);
    collected_kernels.push_back(kernel_idx_to_name(kernel_idx));
  } else {
    printf("Too many arguments! Usage: ./reduce for testing of all kernels; "
           "./reduce [kernel idx] for testing one kernel.");
    exit(EXIT_FAILURE);
  }
  CudaDeviceInfo(); // Print device information.
  run_tests(collected_kernels);

  return 0;
}