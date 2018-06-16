#include <cstdio>
#include <cassert>

int main(int argc, char **argv) {
    int err = 0;
    int n;
    cudaGetDeviceCount(&n);

    for (int i = 0; i < n; ++i) {


        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);


    
        printf("Device %d: %s\n", i, prop.name);

#if __CUDACC_VER_MAJOR__ > 8 && __CUDACC_VER_MINOR__ >  1
        printf("\tcudaDeviceProp.pageableMemoryAccessUsesHostPageTables: %d\n", prop.pageableMemoryAccessUsesHostPageTables);
        printf("\t\tDevice accesses pageable memory using host page tables.\n");
        printf("\t\tThis suggests Address Translation Services are enabled on Power9\n");
#endif

#if __CUDACC_VER_MAJOR__ > 8
        printf("\tcudaDeviceProp.canUseHostPointerForRegisteredMem: %d\n", prop.canUseHostPointerForRegisteredMem);
        printf("\t\tDevice can access host registered memory at the same virtual address as the CPU.\n");
#endif

#if __CUDACC_VER_MAJOR__ > 8 && __CUDACC_VER_MINOR__ >  1
{
        int v;
        cudaDeviceGetAttribute ( &v, cudaDevAttrDirectManagedMemAccessFromHost, i );
        printf("\tcudaDevAttrDirectManagedMemAccessFromHost: %d\n", v);
        printf("\t\tHost can directly access managed memory on the device without migration.\n");
}
#endif

#if __CUDACC_VER_MAJOR__ > 8 && __CUDACC_VER_MINOR__ >  1
{
        int v;
        cudaDeviceGetAttribute ( &v, cudaDevAttrCanFlushRemoteWrites, i );
        printf("\tcudaDevAttrCanFlushRemoteWrites: %d\n", v);
        printf("\t\tdevice supports flushing of outstanding remote writes.\n");
}
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


        {
                cudaSetDevice(i);
                cudaSharedMemConfig config;
                cudaDeviceGetSharedMemConfig ( &config );
                if (cudaSharedMemBankSizeFourByte == config) {
                        printf("\tcudaDeviceGetSharedMemConfig: cudaSharedMemBankSizeFourByte\n");
                } else if ( cudaSharedMemBankSizeEightByte == config) {
                        printf("\tcudaDeviceGetSharedMemConfig: cudaSharedMemBankSizeEightByte\n");
                } else {
                        printf("\tcudaDeviceGetSharedMemConfig: UNKNOWN\n");
                        err = 1;
                }
        }

        {
                cudaSetDevice(i);
                cudaFuncCache config;
                cudaDeviceGetCacheConfig ( &config );
                if (cudaFuncCachePreferNone == config) {
                        printf("\tcudaDeviceGetCacheConfig: cudaFuncCachePreferNone\n");
                        printf("\t\tno preference for shared memory or L1, or sizes are fixed\n");
                } else if ( cudaFuncCachePreferShared == config) {
                        printf("\tcudaDeviceGetCacheConfig: cudaFuncCachePreferShared\n");
                        printf("\t\tprefer larger shared memory and smaller L1 cache\n");
                } else if ( cudaFuncCachePreferL1 == config) {
                        printf("\tcudaDeviceGetCacheConfig: cudaFuncCachePreferL1\n");
                        printf("\t\tprefer larger L1 cache and smaller shared memory\n");
                } else if ( cudaFuncCachePreferEqual == config) {
                        printf("\tcudaDeviceGetCacheConfig: cudaFuncCachePreferEqual\n");
                        printf("\t\tprefer equal size L1 cache and shared memory\n");
                } else {
                        printf("\tcudaDeviceGetCacheConfig: UNKNOWN\n");
                        err = 1;
                }
        }

    }

    return err;
}