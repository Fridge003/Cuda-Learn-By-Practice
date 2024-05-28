## Introduction

When I was self-learning CUDA programming, I found that it is necessary to work under an environment that manages and compiles the kernels in an efficient way. So I built up this repository under the inspiration of several other repositories on github, especially [SGEMM_CUDA](https://github.com/siboehm/SGEMM_CUDA).

## Usage

### Compile

To compile, run the following commands:
```bash
mkdir build
cd build
cmake ..
make
```

### List valid kernels

To display the list of all valid kernels along with their ids:
```bash
./gemm --list-kernels
```

Valid kernels should be registered in the list `registered_kernel` in `gemm/config.h`.


### Test correctness and performance of kernel(s)

In this code repository, we provide features for testing the correctness and performance(latency) of the implemented matrix kernels. These tests need to trigger the kernel for multiple times.

To do these tests, just following the steps below:

First set the sizes of matrices A, B, C in variable `mnk_list` of file `./gemm/config.h`, and compile again. `mnk_list` records all the sets of sizes you want to test.

Then, dependent on whether you want to test the correctness and performance on all kernels or one of them:

To run tests on all GEMM kernels, execute

```bash
DEVICE=[device_id] ./gemm
```
Here the device_id of gpu is 0 by default.

To run tests on one specific kerenel, execute
```bash
DEVICE=[device_id] ./gemm [kernel_idx]
```
Here the kernel_idx should be valid.


### Trigger kernel once without testing

Sometimes we only want to trigger a kernel once. cks. In such cases, you can trigger the kernel once through sending a flag and the sizes of matrices:
```bash
DEVICE=[device_id] ./gemm --once [kernel_idx] [M] [N] [K]
```

For example, when using nsight compute to profile a kernel, just use following command:
```bash
DEVICE=[device_id] ncu -o profile ./gemm --once [kernel_idx] [M] [N] [K]
```


## Benchmark

The following are each kernel's performance of running 8192x8192 GEMM on NVIDIA Tesla V100-PCIE-32GB with 16.4 TFLOPS computing capacity:

<!-- benchmark_results -->
|Idx| Kernel                           |  GFLOPs/s | Performance relative to cuBLAS |
|:--|----------------------------------|----------:|:-------------------------------|
| 1 | Naive                            |   `228.1` | 1.73%                          |
| 2 | Global Memory Coalescing         |  `1774.0` | 13.45%                         |
| 3 | Shared Memory Cache Blocking     |  `3865.2` | 29.30%                         |
| 4 | 1D Block Tiling                  |  `6531.7` | 49.51%                         |
| 5 | 2D Block Tiling                  |  `9409.5` | 71.32%                         |
| 6 | Vectorized 2D Block Tiling       | `11307.8` | 85.71%                         |
| 7 | Double Buffering                 | `11444.1` | 86.74%                         |
| 8 | Bank Conflict Avoiding           | `11780.5` | 89.29%                         |
| 0 | cuBLAS                           | `13193.0` | 100.00%                        |
<!-- benchmark_results -->

## Reference

[SGEMM_CUDA](https://github.com/siboehm/SGEMM_CUDA) by [siboehm](https://github.com/siboehm) (This tutorial is really awesome!)

[Cuda Tutorial](https://cuda.keter.top/) by [PaddleJitLab](https://github.com/PaddleJitLab)

[cuda-sgemm-optimization](https://github.com/YangLinzhuo/cuda-sgemm-optimization) by [yanglinzhuo](https://github.com/YangLinzhuo)

[Cuda Learning Material](https://github.com/ifromeast/cuda_learning.git) by [ifromeast](https://github.com/ifromeast)