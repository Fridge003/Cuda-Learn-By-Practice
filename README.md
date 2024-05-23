## Introduction

When I was self-learning CUDA programming, I found that it is necessary to work under an environment that manages and compiles the kernels in an efficient way. So I built up this repository under the inspiration of several other repositories on github, especially [SGEMM_CUDA](https://github.com/siboehm/SGEMM_CUDA).

## Usage

To compile, execute following commands:
```bash
mkdir build
cd build
cmake ..
make
```

To run test on all gemm kernels:
```bash
DEVICE=[device_id] ./gemm
```
Here the device_id of gpu is 0 by default.

To run a specific gemm kernel(the kernel_idx should be an integer between 0 and kernel number):
```bash
DEVICE=[device_id] ./gemm [kernel_idx]
```

To display the list of implemented kernels:
```bash
./gemm -1
```

The list of implemented kernels are registered in the list `registered_kernel` in `gemm/config.h`.

To change the shapes of matrices for testing, modify the `mnk_list` variable in `gemm/config.h` and compile again.



## Benchmark

The following are each kernel's performance of running 8192x8192 GEMM on NVIDIA Tesla V100-PCIE-32GB with 16.4 TFLOPS computing capacity:

<!-- benchmark_results -->
|Idx| Kernel                           |  GFLOPs/s | Performance relative to cuBLAS |
|:--|----------------------------------|----------:|:-------------------------------|
| 1 | Naive                            |   `228.0` | 1.71%                          |
| 2 | Global Memory Coalescing         |  `1836.1` | 13.77%                         |
| 3 | Shared Memory Cache Blocking     |  `4087.8` | 30.65%                         |
| 4 | 1D Block Tiling                  |  `6748.6` | 50.59%                         |
| 5 | 2D Block Tiling                  |  `9799.3` | 73.46%                         |
| 6 | Vectorized 2D Block Tiling       | `11712.5` | 87.81%                         |
| 0 | cuBLAS                           | `13338.9` | 100.00%                        |
<!-- benchmark_results -->

## Reference

[SGEMM_CUDA](https://github.com/siboehm/SGEMM_CUDA) by [siboehm](https://github.com/siboehm) (This tutorial is really awesome!)

[Cuda Tutorial](https://cuda.keter.top/) by [PaddleJitLab](https://github.com/PaddleJitLab)

[cuda-sgemm-optimization](https://github.com/YangLinzhuo/cuda-sgemm-optimization) by [yanglinzhuo](https://github.com/YangLinzhuo)

[Cuda Learning Material](https://github.com/ifromeast/cuda_learning.git) by [ifromeast](https://github.com/ifromeast)


