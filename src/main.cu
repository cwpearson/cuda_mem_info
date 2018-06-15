#include <cstdio>

int main(int argc, char **argv) {

    int n;
    cudaGetDeviceCount(&n);

    for (int i = 0; i < n; ++i) {


        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
    
        printf("Device %d: %s\n", i, prop.name);

#if __CUDACC_VER_MAJOR__ > 8
        printf("\tcudaDeviceProp.pageableMemoryAccessUsesHostPageTables: %d\n", prop.pageableMemoryAccessUsesHostPageTables);
        printf("\t\tDevice accesses pageable memory using host page tables.\n");
        printf("\t\tThis suggests Address Translation Services are enabled on Power9\n");
#endif

#if __CUDACC_VER_MAJOR__ > 8
        printf("\tcudaDeviceProp.canUseHostPointerForRegisteredMem: %d\n", prop.canUseHostPointerForRegisteredMem);
        printf("\t\tDevice can access host registered memory at the same virtual address as the CPU.\n");
#endif

#if __CUDACC_VER_MAJOR__ > 8
        int v;
        cudaDeviceGetAttribute ( &v, cudaDevAttrDirectManagedMemAccessFromHost, i );
        printf("\tcudaDevAttrDirectManagedMemAccessFromHost: %d\n", v);
        printf("\t\tHost can directly access managed memory on the device without migration.\n");
#endif

        printf("\tcudaDeviceProp.pageableMemoryAccess: %d\n", prop.pageableMemoryAccess);
        printf("\t\tDevice supports coherently accessing pageable memory without calling cudaHostRegister on it.\n");

        printf("\tcudaDeviceProp.concurrentManagedAccess: %d\n", prop.concurrentManagedAccess);
        printf("\t\tDevice can coherently access managed memory concurrently with the CPU.\n");

        printf("\tcudaDeviceProp.canMapHostMemory: %d\n", prop.canMapHostMemory);
        printf("\t\tDevice can map host memory into the CUDA address space for use with cudaHostAlloc()/cudaHostGetDevicePointer().\n");

        printf("\tcudaDeviceProp.totalGlobalMem: %lu\n", prop.totalGlobalMem);
        printf("\t\tbytes\n");

        printf("\tcudaDeviceProp.totalConstMem: %lu\n", prop.totalConstMem);
        printf("\t\tbytes\n");

        printf("\tcudaDeviceProp.sharedMemPerBlock: %lu\n", prop.sharedMemPerBlock);
        printf("\t\tbytes\n");

        printf("\tcudaDeviceProp.sharedMemPerMultiprocessor: %lu\n", prop.sharedMemPerMultiprocessor);
        printf("\t\tbytes\n");

        printf("\tcudaDeviceProp.asyncEngineCount: %d\n", prop.asyncEngineCount);
        printf("\t\t1 when the device can concurrently copy memory between host and device while executing a kernel.\n"
        "\t\t2 when the device can concurrently copy memory between host and device in both directions and execute a kernel at the same time.\n");

        printf("\tcudaDeviceProp.globalL1CacheSupported: %d\n", prop.globalL1CacheSupported);
        printf("\t\tDevice supports caching of globals in L1 cache.\n");

        printf("\tcudaDeviceProp.localL1CacheSupported: %d\n", prop.localL1CacheSupported);
        printf("\t\tDevice supports caching of locals in L1 cache.\n");

        printf("\tcudaDeviceProp.l2CacheSize: %d\n", prop.l2CacheSize);
        printf("\t\tbytes\n");

        printf("\tcudaDeviceProp.memoryBusWidth: %d\n", prop.memoryBusWidth);
        printf("\t\tbits\n");

        printf("\tcudaDeviceProp.memoryClockRate: %d\n", prop.memoryClockRate);
        printf("\t\tkhz\n");
    }

    return 0;
}