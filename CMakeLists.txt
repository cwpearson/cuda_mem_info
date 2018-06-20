cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(cmake_and_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_EXTENSIONS OFF)
set(VERBOSE_BUILD ON)
include_directories(SYSTEM "thirdparty/cxxopts/include")
set (CUDA_SEPARABLE_COMPILATION ON)

# Should be a FindNvidiaUvm.cmake module
# include_directories("/usr/src/nvidia-396.24.02")
# include_directories("/usr/src/nvidia-396.24.02/common/inc")
# add_definitions(-D__KERNEL__)

add_subdirectory(src)
