#pragma once

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "runner.cuh"

#define CUDA_CHECK(err) (CudaCheck(err, __FILE__, __LINE__))

void print_border_line() {
  for (int i = 0; i < 30; ++i) {
    printf("-");
  }
  printf("\n");
}

void CudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
};

void CudaDeviceInfo() {
  print_border_line();

  int deviceId;
  cudaGetDevice(&deviceId);

  cudaDeviceProp props{};
  cudaGetDeviceProperties(&props, deviceId);

  printf("Device ID: %d\n\
    Name: %s\n\
    Compute Capability: %d.%d\n\
    memoryBusWidth: %d\n\
    maxThreadsPerBlock: %d\n\
    maxThreadsPerMultiProcessor: %d\n\
    maxRegsPerBlock: %d\n\
    maxRegsPerMultiProcessor: %d\n\
    totalGlobalMem: %zuMB\n\
    sharedMemPerBlock: %zuKB\n\
    sharedMemPerMultiprocessor: %zuKB\n\
    totalConstMem: %zuKB\n\
    multiProcessorCount: %d\n\
    Warp Size: %d\n",
         deviceId, props.name, props.major, props.minor, props.memoryBusWidth,
         props.maxThreadsPerBlock, props.maxThreadsPerMultiProcessor,
         props.regsPerBlock, props.regsPerMultiprocessor,
         props.totalGlobalMem / 1024 / 1024, props.sharedMemPerBlock / 1024,
         props.sharedMemPerMultiprocessor / 1024, props.totalConstMem / 1024,
         props.multiProcessorCount, props.warpSize);
}

void randomize_array(float *arr, int N) {
  srand(time(0));
  for (int i = 0; i < N; i++) {
    float tmp = (float)(rand() % 5);
    tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
    arr[i] = tmp;
  }
}

void zero_init_array(float *arr, int N) { memset(arr, 0, N * sizeof(float)); }

bool check_result_correctness(float *arr, double sum_ref, int arr_len) {
  bool correct = true;
  double sum = 0.0;
  double eps = 1e-6;

  for (int i = 0; i < arr_len; i++) {
    sum += arr[i];
  }

  if (abs(sum - sum_ref) > eps) {
    correct = false;
    printf("Correctness Check: Not Pass!");
  } else {
    printf("Correctness Check: Pass!\n");
  }

  return correct;
}

void check_performance(const std::string &kernel, float *d_in, float *d_out,
                       int N, int block_size, int warmup_num = 10,
                       int profile_num = 10) {

  float total_running_time = 0.0;
  float current_running_time = 0.0;

  // Do warmup.
  for (int j = 0; j < warmup_num; ++j) {
    run_kernel(d_in, d_out, N, block_size, kernel);
  }

  // Profile kernel.
  for (int j = 0; j < profile_num; ++j) {
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);

    run_kernel(d_in, d_out, N, block_size, kernel);
    cudaDeviceSynchronize();

    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);

    cudaEventElapsedTime(&current_running_time, start, end);
    total_running_time += current_running_time;
  }

  double avg_latency = total_running_time / profile_num;
  double avg_bandwidth =
      (double(N)) * 4 / 1024 / 1024 / 1024 / avg_latency * 1000;

  // Print Result.
  printf("AVG Latency = %12.8lf ms, AVG Bandwidth = %10.8lf GB/s\n",
         avg_latency, avg_bandwidth);
}