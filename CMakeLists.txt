cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cmake_and_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_EXTENSIONS OFF)
include_directories("thirdparty/cxxopts/include")

add_subdirectory(src)
