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
Device 0: GeForce GTX 1070
        cudaDeviceProp.pageableMemoryAccessUsesHostPageTables: 0
                Device accesses pageable memory using host page tables.
                This suggests Address Translation Services are enabled on Power9
        cudaDeviceProp.canUseHostPointerForRegisteredMem: 1
                Device can access host registered memory at the same virtual address as the CPU.
        cudaDevAttrDirectManagedMemAccessFromHost: 0
                Host can directly access managed memory on the device without migration.
        cudaDeviceProp.pageableMemoryAccess: 0
                Device supports coherently accessing pageable memory without calling cudaHostRegister on it.
        cudaDeviceProp.concurrentManagedAccess: 1
                Device can coherently access managed memory concurrently with the CPU.
        cudaDeviceProp.canMapHostMemory: 1
                Device can map host memory into the CUDA address space for use with cudaHostAlloc()/cudaHostGetDevicePointer().
        cudaDeviceProp.totalGlobalMem: 8510701568
                bytes
        cudaDeviceProp.totalConstMem: 65536
                bytes
        cudaDeviceProp.sharedMemPerBlock: 49152
                bytes
        cudaDeviceProp.sharedMemPerMultiprocessor: 98304
                bytes
        cudaDeviceProp.asyncEngineCount: 2
                1 when the device can concurrently copy memory between host and device while executing a kernel.
                2 when the device can concurrently copy memory between host and device in both directions and execute a kernel at the same time.
        cudaDeviceProp.globalL1CacheSupported: 1
                Device supports caching of globals in L1 cache.
        cudaDeviceProp.localL1CacheSupported: 1
                Device supports caching of locals in L1 cache.
        cudaDeviceProp.l2CacheSize: 2097152
                bytes
        cudaDeviceProp.memoryBusWidth: 256
                bits
        cudaDeviceProp.memoryClockRate: 4004000
                khz
```
