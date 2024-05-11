#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#include "runner.cuh"

void print_border_line() {
    for (int i = 0; i < 30; ++i) {
        printf("-");
    }
    printf("\n");
}

void range_init_matrix(float* mat, int N) {
    for (int i = 0; i < N; i++) {
        mat[i] = i;
    }
}

void randomize_matrix(float* mat, int N) {
    srand(time(0));
    for (int i = 0; i < N; i++) {
        float tmp = (float)(rand() % 5);
        tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
        mat[i] = tmp;
    }
}

void zero_init_matrix(float* mat, int N) {
    memset(mat, 0, N * sizeof(float));
}

void print_matrix(float* A, int M, int N){
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++){
           printf("%f ", A[i * N + j]);
        }
        printf("\n");
    }
}

void estimate_compute_and_IO_cost(int M, int N, int K, double compute_capacity, double bandwidth) {
    // compute_capacity: fp32 computing ability of GPU in TFLOPS
    // bandwidth: bandwidth between GPU global memory and chip, in GB/s
    double total_flops = (double(M)) * N * K * 2;
    double total_data_IO_memory = ((M * K) + (K * N) + (M * N)) * 4;  // float dtype has 4 bytes
    
    printf("Amount of computation: %lf GFLOPS; Amount of memory IO: %lf MB\n", 
            total_flops / (1024 * 1024 * 1024), 
            total_data_IO_memory / (1024 * 1024));

    printf("Theoretical computation time: %lf ms; Theoretical IO time: %lf ms; Ratio of Compute to IO: %lf\n", 
            total_flops * 1000 / (compute_capacity * 1024 * 1024 * 1024 * 1024),
            total_data_IO_memory * 1000 / (bandwidth * 1024 * 1024 * 1024),
            (total_flops * bandwidth)/ (1024 * total_data_IO_memory * compute_capacity));
}


void sync_device_and_check_kernel_error() {
    cudaError_t errSync = cudaGetLastError();                             
    cudaError_t errAsync = cudaDeviceSynchronize();                       
    if (errSync != cudaSuccess) {
        printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
        exit(EXIT_FAILURE);
    }

    if (errAsync != cudaSuccess) {
        printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
        exit(EXIT_FAILURE);
    }    
}

bool check_result_correctness(float* mat, float* mat_ref, int M, int N) {
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
        } if (err > eps) {
            incorrect_cnt++;
            max_err = max(max_err, err);
        }
    }

    if (nan_cnt > 0 || incorrect_cnt > 0) {
        correct = false;
        printf("Correctness Check: Not Pass! Incorrect elements: %d/%d, NaN elements: %d/%d, Max Error: %f\n", incorrect_cnt, total_cnt, nan_cnt, total_cnt, max_err);
    } else {
        printf("Correctness Check: Pass!\n");
    }

    return correct;
}


void check_performance(const std::string& kernel,  float* d_A, float* d_B, float* d_C, 
                       int M, int N, int K, int repeat_num = 10) {
                
    float total_running_time = 0.0;
    float current_running_time = 0.0;

    for (int j = 0; j < repeat_num; j++) {
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

    double avg_latency = total_running_time / repeat_num;
    double avg_Gflops = (double(M)) * N * K * 2 / 1024 / 1024 / 1024 / avg_latency * 1000;

    // Print Result.
    printf("AVG Latency = %12.8lf ms, AVG Performance = %10.8lf Gflops\n", avg_latency, avg_Gflops);
}