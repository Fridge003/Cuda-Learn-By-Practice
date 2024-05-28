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

void range_init_matrix(float *mat, int N) {
  for (int i = 0; i < N; i++) {
    mat[i] = i;
  }
}

void randomize_matrix(float *mat, int N) {
  srand(time(0));
  for (int i = 0; i < N; i++) {
    float tmp = (float)(rand() % 5);
    tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
    mat[i] = tmp;
  }
}

void zero_init_matrix(float *mat, int N) { memset(mat, 0, N * sizeof(float)); }

void print_matrix(float *A, int M, int N) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      printf("%f ", A[i * N + j]);
    }
    printf("\n");
  }
}

void estimate_compute_and_IO_cost(int M, int N, int K, double compute_capacity,
                                  double bandwidth) {
  // compute_capacity: fp32 computing ability of GPU in TFLOPS
  // bandwidth: bandwidth between GPU global memory and chip, in GB/s
  double total_flops = (double(M)) * N * K * 2;
  double total_data_IO_memory =
      (((double)(M)*K) + (double(K) * N) + (double(M) * N)) *
      4; // float dtype has 4 bytes

  printf("Amount of computation: %lf GFLOPS; Amount of memory IO: %lf MB\n",
         total_flops / (1024 * 1024 * 1024),
         total_data_IO_memory / (1024 * 1024));

  printf("Theoretical computation time: %lf ms; Theoretical IO time: %lf ms; "
         "Ratio of Compute to IO: %lf\n",
         total_flops * 1000 / (compute_capacity * 1024 * 1024 * 1024 * 1024),
         total_data_IO_memory * 1000 / (bandwidth * 1024 * 1024 * 1024),
         (total_flops * bandwidth) /
             (1024 * total_data_IO_memory * compute_capacity));
}

bool check_result_correctness(float *mat, float *mat_ref, int M, int N) {
  bool correct = true;
  float eps = 1e-6;
  int nan_cnt = 0;
  int incorrect_cnt = 0;
  int total_cnt = M * N;
  float max_err = 0;

  for (int i = 0; i < total_cnt; i++) {
    float err = abs(mat[i] - mat_ref[i]);
    if (err != err) { // NaN
      nan_cnt++;
    }
    if (err > eps) {
      incorrect_cnt++;
      max_err = max(max_err, err);
    }
  }

  if (nan_cnt > 0 || incorrect_cnt > 0) {
    correct = false;
    printf("Correctness Check: Not Pass! Incorrect elements: %d/%d, NaN "
           "elements: %d/%d, Max Error: %f\n",
           incorrect_cnt, total_cnt, nan_cnt, total_cnt, max_err);
  } else {
    printf("Correctness Check: Pass!\n");
  }

  return correct;
}

void check_performance(const std::string &kernel, float *d_A, float *d_B,
                       float *d_C, int M, int N, int K, int warmup_num = 10,
                       int profile_num = 10) {

  float total_running_time = 0.0;
  float current_running_time = 0.0;

  // Do warmup.
  for (int j = 0; j < warmup_num; ++j) {
    run_kernel(d_A, d_B, d_C, M, N, K, kernel);
  }

  // Profile kernel.
  for (int j = 0; j < profile_num; ++j) {
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);

    run_kernel(d_A, d_B, d_C, M, N, K, kernel);
    cudaDeviceSynchronize();

    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);

    cudaEventElapsedTime(&current_running_time, start, end);
    total_running_time += current_running_time;
  }

  double avg_latency = total_running_time / profile_num;
  double avg_Gflops =
      (double(M)) * N * K * 2 / 1024 / 1024 / 1024 / avg_latency * 1000;

  // Print Result.
  printf("AVG Latency = %12.8lf ms, AVG Performance = %10.8lf Gflops\n",
         avg_latency, avg_Gflops);
}