cmake_minimum_required(VERSION 3.18)
project(cuda_learning LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)

# configure Cuda environments
find_package(CUDA REQUIRED)
include(CheckLanguage)
check_language(CUDA)
set(CUDA_COMPUTE_CAPABILITY 70) # 7.0 is suitable for V100, change for higher version if using newer GPUs
include_directories(${CUDA_INCLUDE_DIRS})

# generate executable for GEMM kernels
aux_source_directory(${PROJECT_SOURCE_DIR}/gemm GEMM_SRC)
add_executable(gemm ${PROJECT_SOURCE_DIR}/gemm/gemm.cu ${GEMM_SRC})
set_target_properties(gemm PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_link_libraries(gemm ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

# generate executable for reduce kernels
# aux_source_directory(${PROJECT_SOURCE_DIR}/reduce REDUCE_SRC)
# add_executable(reduce ${PROJECT_SOURCE_DIR}/reduce/reduce.cu ${REDUCE_SRC})
# set_target_properties(reduce PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
# target_link_libraries(reduce ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})