This repository stores codes when I was learning CUDA programming.

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

To run a specific gemm kernel:
```bash
./gemm [kernel_idx]
```

The following are each kernel's performance of running 4096x4096 GEMM on NVIDIA V100:

<!-- benchmark_results -->
|Idx| Kernel                           |  GFLOPs/s | Performance relative to cuBLAS |
|:--|----------------------------------|----------:|:-------------------------------|
| 1 | Naive                            |   `220.0` | 1.96%                          |
| 2 | Global Memory Coalescing         |  `1919.3` | 17.09%                         |
| 3 | Shared Memory Cache Blocking     |  `3986.6` | 35.51%                         |
| 0 | cuBLAS                           | `11227.6` | 100.0%                         |
<!-- benchmark_results -->

## Reference

[SGEMM_CUDA](https://github.com/siboehm/SGEMM_CUDA) by [siboehm](https://github.com/siboehm) (This tutorial is really awesome!)

[Cuda Tutorial](https://cuda.keter.top/) by [PaddleJitLab](https://github.com/PaddleJitLab)

[UNIVERSAL_SGEMM_CUDA](https://github.com/AndSonder/UNIVERSAL_SGEMM_CUDA) by [AndSonder](https://github.com/AndSonder)

[Cuda Learning Material](https://github.com/ifromeast/cuda_learning.git) by [ifromeast](https://github.com/ifromeast)


