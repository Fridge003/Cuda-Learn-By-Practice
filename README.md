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
./gemm
```

To run a specific gemm kernel(the kernel_idx should be an integer between 0 and kernel number):
```bash
./gemm [kernel_idx]
```

To display the list of implemented kernels:
```bash
./gemm -1
```

The list of implemented kernels are registered in the list `registered_kernel` in `gemm/config.h`.

To change the shapes of matrices for testing, modify the `mnk_list` variable in `gemm/config.h` and compile again.



## Benchmark

The following are each kernel's performance of running 4096x4096 GEMM on NVIDIA Tesla V100-SXM2-16GB:

<!-- benchmark_results -->
|Idx| Kernel                           |  GFLOPs/s | Performance relative to cuBLAS |
|:--|----------------------------------|----------:|:-------------------------------|
| 1 | Naive                            |   `220.0` | 1.96%                          |
| 2 | Global Memory Coalescing         |  `1919.3` | 17.09%                         |
| 3 | Shared Memory Cache Blocking     |  `3986.6` | 35.51%                         |
| 4 | 1D Block Tiling                  |  `6589.1` | 58.69%                         |
| 0 | cuBLAS                           | `11227.6` | 100.00%                         |
<!-- benchmark_results -->

## Reference

[SGEMM_CUDA](https://github.com/siboehm/SGEMM_CUDA) by [siboehm](https://github.com/siboehm) (This tutorial is really awesome!)

[Cuda Tutorial](https://cuda.keter.top/) by [PaddleJitLab](https://github.com/PaddleJitLab)

[UNIVERSAL_SGEMM_CUDA](https://github.com/AndSonder/UNIVERSAL_SGEMM_CUDA) by [AndSonder](https://github.com/AndSonder)

[Cuda Learning Material](https://github.com/ifromeast/cuda_learning.git) by [ifromeast](https://github.com/ifromeast)


