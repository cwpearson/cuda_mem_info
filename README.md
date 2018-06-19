# cuda_mem_info

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

  -f, --format shell|csv|md  Output Format (default: shell)
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
$src/info -f md
|cudaDeviceProp.pageableMemoryAccessUsesHostPageTables|0|
|-|-|
|cudaDeviceProp.canUseHostPointerForRegisteredMem|1|
|cudaDevAttrDirectManagedMemAccessFromHost|0|
|cudaDevAttrCanFlushRemoteWrites|0|
|cudaDeviceProp.pageableMemoryAccess|0|
|cudaDeviceProp.concurrentManagedAccess|1|
|cudaDeviceProp.canMapHostMemory|1|
|cudaDeviceProp.totalGlobalMem|8510701568|
|cudaDeviceProp.totalConstMem|65536|
|cudaDeviceProp.sharedMemPerBlock|49152|
|cudaDeviceProp.sharedMemPerMultiprocessor|98304|
|cudaDeviceProp.l2CacheSize|2097152|
|cudaDeviceProp.memoryBusWidth|256|
|cudaDeviceProp.memoryClockRate|4004000|
|cudaDeviceProp.asyncEngineCount|2|
|cudaDeviceProp.globalL1CacheSupported|1|
|cudaDeviceProp.localL1CacheSupported|1|
```