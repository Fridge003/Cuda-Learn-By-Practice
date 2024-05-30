#pragma once

#include <string>
#include <unordered_map>
#include <vector>

// List of m, n and k values that are supposed to be tested.
const std::vector<std::vector<int>> mnk_list = {
    {128, 128, 128},    {256, 256, 256},    {512, 512, 512},
    {1024, 1024, 1024}, {2048, 2048, 2048}, {4096, 4096, 4096},
    {8192, 8192, 8192}};

// List of kernels that have been implemented and are supposed to be tested.
const std::vector<std::string> registered_kernel = {
    "cublas",
    "naive",
    "global_memory_coalescing",
    "shared_memory_cache_blocking",
    "1D_block_tiling",
    "2D_block_tiling",
    "vectorized_2D_block_tiling",
    "double_buffering",
    "bank_conflict_avoiding",
    "warp_tiling"};

// Parameters of V100 for PCIe
// const double device_fp32_compute_capacity_tflops = 14.0;
// const double device_global_mem_bandwidth_GB_per_sec = 900.0;

// Parameters of V100S for PCIe
const double device_fp32_compute_capacity_tflops = 16.4;
const double device_global_mem_bandwidth_GB_per_sec = 1134.0;

// Parameters of A100 for PCIe
// const double device_fp32_compute_capacity_tflops = 19.5;
// const double device_global_bandwidth_GB_per_sec = 1935.0;
