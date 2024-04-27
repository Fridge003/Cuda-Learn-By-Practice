cmake_minimum_required(VERSION 3.18)
project(cuda_learning LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)

find_package(CUDA REQUIRED)

# ensure cuda is available
include(CheckLanguage)
check_language(CUDA)
set(CUDA_COMPUTE_CAPABILITY 70) # 7.0 is suitable for V100, change for higher version if using newer GPUs

# in debug mode, add debug symbols to device code
# this disables most optimizations and kills performance
add_compile_options("$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")

# Configure header file search paths
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/gemm)

# Configure the source file path to be compiled
aux_source_directory(${PROJECT_SOURCE_DIR}/gemm GEMM_SRC)

# generate executable
add_executable(gemm ${PROJECT_SOURCE_DIR}/gemm/gemm.cu ${GEMM_SRC})
set_target_properties(gemm PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_link_libraries(gemm ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})