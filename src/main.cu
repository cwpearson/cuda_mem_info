#include <cstdio>

int main(int argc, char **argv) {

    int n;
    cudaGetDeviceCount(&n);

    for (int i = 0; i < n; ++i) {
        printf("Device %d\n", i);

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
    
        printf("\tcudaDeviceProp.pageableMemoryAccessUsesHostPageTables: %d\n", prop.pageableMemoryAccessUsesHostPageTables);
        printf("\t\tThis device accesses pageable memory using host page tables.\n");
        printf("\t\tThis suggests Address Translation Services are enabled on Power9\n");
    
        int v;
        cudaDeviceGetAttribute ( &v, cudaDevAttrDirectManagedMemAccessFromHost, i );

        printf("\tcudaDevAttrDirectManagedMemAccessFromHost: %d\n", v);
        printf("\t\tHost can directly access managed memory on the device without migration.\n");
    }

    return 0;
}