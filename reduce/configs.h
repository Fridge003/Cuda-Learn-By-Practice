#pragma once

#include <string>
#include <unordered_map>
#include <vector>

// Number of threads per block.
const int block_size = 512;

// Number of elements summed by each thread before doing blockwise reduction.
const int num_per_thread = 4;

// List of n(length of reduced array) to be tested.
const std::vector<int> n_list = {100000000};

// List of kernels that have been implemented and are supposed to be tested.
const std::vector<std::string> registered_kernel = {
    "baseline", "shared_memory", "multiple_add", "warp_unrolling",
    "warp_shuffle_down"};

// Bandwidth of different GPUs
// const double device_global_mem_bandwidth_GB_per_sec = 900.0; // V100 PCIe
const double device_global_mem_bandwidth_GB_per_sec = 1134.0; // V100S PCIe
// const double device_global_bandwidth_GB_per_sec = 1935.0; // A100 PCIe
