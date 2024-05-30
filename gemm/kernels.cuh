#pragma once

#include "kernels/1D_block_tiling.cuh"
#include "kernels/2D_block_tiling.cuh"
#include "kernels/bank_conflict_avoiding.cuh"
#include "kernels/double_buffering.cuh"
#include "kernels/global_memory_coalescing.cuh"
#include "kernels/naive.cuh"
#include "kernels/shared_memory_cache_blocking.cuh"
#include "kernels/vectorized_2D_block_tiling.cuh"
#include "kernels/warp_tiling.cuh"