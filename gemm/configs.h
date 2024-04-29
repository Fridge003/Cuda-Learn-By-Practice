#pragma once

#include <vector>
#include <string>

// List of m, n and k values that are supposed to be tested.
const std::vector<int> m_list = {128, 256, 512, 1024, 2048};
const std::vector<int> n_list = {128, 256, 512, 1024, 2048};
const std::vector<int> k_list = {128, 256, 512, 1024, 2048};

// const std::vector<int> m_list = {2, 8, 32, 128, 2, 8, 32, 128};
// const std::vector<int> n_list = {2048, 2048, 2048, 2048, 4096, 4096, 4096, 4096};
// const std::vector<int> k_list = {2048, 2048, 2048, 2048, 4096, 4096, 4096, 4096};

// List of kernels that have been implemented and are supposed to be tested.
const std::vector<std::string> kernel_list = {"naive", "cublas", "global_memory_coalescing"};

// Parameters of V100 for PCIe
// const double device_fp32_compute_capacity_tflops = 14.0;
// const double device_global_mem_bandwidth_GB_per_sec = 900.0;

// Parameters of V100S for PCIe
const double device_fp32_compute_capacity_tflops = 16.4;
const double device_global_mem_bandwidth_GB_per_sec = 1134.0;

// Parameters of A100 for PCIe
// const double device_fp32_compute_capacity_tflops = 19.5;
// const double device_global_bandwidth_GB_per_sec = 1935.0;

