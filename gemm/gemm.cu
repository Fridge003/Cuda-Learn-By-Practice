#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <cassert>
#include <cuda_runtime.h>

#include "runner.cuh"
#include "utils.cuh"

const std::vector<int> m_list = {128, 256, 512, 1024, 2048};
const std::vector<int> n_list = {128, 256, 512, 1024, 2048};
const std::vector<int> k_list = {128, 256, 512, 1024, 2048};

// const std::vector<int> m_list = {2, 8, 32, 128, 2, 8, 32, 128};
// const std::vector<int> n_list = {2048, 2048, 2048, 2048, 4096, 4096, 4096, 4096};
// const std::vector<int> k_list = {2048, 2048, 2048, 2048, 4096, 4096, 4096, 4096};

// List of kernels that have been implemented and are supposed to be tested.
const std::vector<std::string> kernel_list = {"naive", "cublas"};

int main(void) {
    assert((m_list.size() == n_list.size()) && (m_list.size() == k_list.size()));
    
    for (int test_case = 0; test_case < m_list.size(); ++test_case) {
        print_border_line();

        const int m = m_list[test_case];
        const int n = n_list[test_case];
        const int k = k_list[test_case];
        printf("Test %d: m = %d, n = %d, k = %d\n", test_case, m, n, k);

        // Allocate space for the matrices, the goal is to calculate A @ B and store it in C
        // h for host, d for device; h_C_ref is for correctness checking
        size_t size_A = m * k * sizeof(float);
        size_t size_B = k * n * sizeof(float);
        size_t size_C = m * n * sizeof(float);
        
        float *h_A, *h_B, *h_C, *h_C_ref, *d_A, *d_B, *d_C, *d_C_ref;
        h_A = (float *)malloc(size_A);
        h_B = (float *)malloc(size_B);
        h_C = (float *)malloc(size_C);
        h_C_ref = (float *)malloc(size_C);
        cudaMalloc(&d_A, size_A);
        cudaMalloc(&d_B, size_B);
        cudaMalloc(&d_C, size_C);
        cudaMalloc(&d_C_ref, size_C);

        // Initialize the matrices, and copy them to device.
        randomize_matrix(h_A, m * k);
        randomize_matrix(h_B, k * n);
        zero_init_matrix(h_C_ref, m * n);
        cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
        cudaMemcpy(d_C_ref, h_C_ref, size_C, cudaMemcpyHostToDevice);

        // Store the correct result in d_C_ref, and copy back.
        run_cublas_gemm(d_A, d_B, d_C_ref, m, n, k);
        cudaMemcpy(h_C_ref, d_C_ref, size_C, cudaMemcpyDeviceToHost);

        // Test each kernel.
        for (const std::string& kernel: kernel_list) {
            printf("\nKernel: %s\n", kernel.c_str());

            zero_init_matrix(h_C, m * n);
            cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice);
            
            // Check kernel validity.
            bool valid_kernel = run_kernel(d_A, d_B, d_C, m, n, k, kernel);
            if (!valid_kernel) continue;

            // Check Correctness.
            sync_device_and_check_kernel_error();
            cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
            bool correct = check_result_correctness(h_C, h_C_ref, m, n);
            if (!correct) continue;

            // Check Performance.
            check_performance(kernel, d_A, d_B, d_C, m, n, k, 5);
        }

        // Free Memory.
        free(h_A);
        free(h_B);
        free(h_C);
        free(h_C_ref);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

    }
    print_border_line();

    return 0;
}