# cuda_mem_info

    cmake -DCMAKE_CUDA_HOST_COMPILER=`which g++-6` .. -DCMAKE_C_COMPILER=gcc-6 -DCMAKE_CXX_COMPILER=g++-6

Get info about CUDA Device memory capabilities

    mkdir build
    cd build
    cmake ..
    make

Run with

    src/info

Sample Output

```
$ src/info -h
Format CUDA device info
Usage:
  src/info [OPTION...]

  -f, --format ascii|csv|md  Output Format (default: ascii)
  -d, --descriptions         Print Descriptions
  -h, --help                 Print help
```

```
$ src/info
Device 0: GeForce GTX 1070
+-------------------------------------------------------+------------+
|                                              Property |      Value |
+-------------------------------------------------------+------------+
| cudaDeviceProp.pageableMemoryAccessUsesHostPageTables |          0 |
|      cudaDeviceProp.canUseHostPointerForRegisteredMem |          1 |
|             cudaDevAttrDirectManagedMemAccessFromHost |          0 |
|                       cudaDevAttrCanFlushRemoteWrites |          0 |
|                   cudaDeviceProp.pageableMemoryAccess |          0 |
|                cudaDeviceProp.concurrentManagedAccess |          1 |
|                       cudaDeviceProp.canMapHostMemory |          1 |
|                         cudaDeviceProp.totalGlobalMem | 8510701568 |
|                          cudaDeviceProp.totalConstMem |      65536 |
|                      cudaDeviceProp.sharedMemPerBlock |      49152 |
|             cudaDeviceProp.sharedMemPerMultiprocessor |      98304 |
|                            cudaDeviceProp.l2CacheSize |    2097152 |
|                         cudaDeviceProp.memoryBusWidth |        256 |
|                        cudaDeviceProp.memoryClockRate |    4004000 |
|                       cudaDeviceProp.asyncEngineCount |          2 |
|                 cudaDeviceProp.globalL1CacheSupported |          1 |
|                  cudaDeviceProp.localL1CacheSupported |          1 |
+-------------------------------------------------------+------------+
```

```
$ src/info -f md
#CUDA Properties
## Device Properties
**Device 0: GeForce GTX 1060 3GB**
|cudaDeviceProp.canUseHostPointerForRegisteredMem|1|
|-|-|
|cudaDeviceProp.pageableMemoryAccess|0|
|cudaDeviceProp.concurrentManagedAccess|1|
|cudaDeviceProp.canMapHostMemory|1|
|cudaDeviceProp.totalGlobalMem|3142516736|
|cudaDeviceProp.totalConstMem|65536|
|cudaDeviceProp.sharedMemPerBlock|49152|
|cudaDeviceProp.sharedMemPerMultiprocessor|98304|
|cudaDeviceProp.l2CacheSize|1572864|
|cudaDeviceProp.memoryBusWidth|192|
|cudaDeviceProp.memoryClockRate|4004000|
|cudaDeviceProp.asyncEngineCount|2|
|cudaDeviceProp.globalL1CacheSupported|1|
|cudaDeviceProp.localL1CacheSupported|1|
```
