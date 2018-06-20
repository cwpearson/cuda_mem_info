#include <cstdio>
#include <cassert>

#include "cxxopts.hpp"

#include "table.hpp"
#include "report.hpp"



//for ioctls
// #include <sys/ioctl.h>
// #include <nvidia-uvm/uvm_ioctl.h>
// #include <nv.h>
// for files
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// #undef XAPIGEN
// #undef NV_STATUS_CODE
// #define NV_STATUS_CODE( name, code, string ) { code,  string " [" #name "]" },

// static struct NvStatusCodeString
// {
//     NV_STATUS   statusCode;
//     const char *statusString;
// } g_StatusCodeList[] = {
//    #include <nvstatuscodes.h>
//    { 0xffffffff, "Unknown error code!" } // Some compilers don't like the trailing ','
// };

// const char *nvstatusToString(NV_STATUS nvStatusIn)
// {
//     NvU32 i;
//     NvU32 n = ((NvU32)(sizeof(g_StatusCodeList))/(NvU32)(sizeof(g_StatusCodeList[0])));
//     printf("%d\n", n);
//     for (i = 0; i < 10; i++)
//     {
//         printf("%d\n", g_StatusCodeList[i].statusCode);
//         if (g_StatusCodeList[i].statusCode == nvStatusIn)
//         {
//             return g_StatusCodeList[i].statusString;
//         }
//     }

//     return "Unknown error code!";
// }

int main(int argc, char **argv) {

        std::string output_format;
        bool print_descriptions;

        Report report("CUDA Properties");

        try
        {
          cxxopts::Options options(argv[0], "Format CUDA device info");
          options
            .positional_help("[optional args]")
            .show_positional_help();
      
          options
        //     .allow_unrecognised_options()
            .add_options()
            ("f,format", "Output Format", cxxopts::value<std::string>(output_format)->default_value("shell"), "shell|csv|md")
            ("d,descriptions", "Print Descriptions", cxxopts::value<bool>(print_descriptions)->default_value("false"))
            ("h,help", "Print help")
          ;
      
          auto result = options.parse(argc, argv);
      
          if (result.count("help"))
          {
            std::cout << options.help({"", "Group"}) << std::endl;
            exit(0);
          }
      
      
        } catch (const cxxopts::OptionException& e)
        {
          std::cout << "error parsing options: " << e.what() << std::endl;
          exit(1);
      }

      /*
      auto path = "/dev/nvidia-uvm";
      auto fd = open(path, O_RDWR);
      if (fd == -1)
        {
        perror("open");
        return 2;
        }

      UVM_PAGEABLE_MEM_ACCESS_PARAMS q;

      if (ioctl(fd, UVM_PAGEABLE_MEM_ACCESS, &q) == -1)
      {
          perror("ioctl get");
      }
      else
      {
          printf("pageableMemAccess : %d\n", q.pageableMemAccess);
          printf("rmStatus: %d\n", q.rmStatus);
      }
      close(fd);

      fd = open("/dev/nvidia0", O_RDWR);
      if (fd == -1)
        {
        perror("open");
        return 2;
        }
    */

/*
      nv_ioctl_numa_info_t numa_info;
      numa_info.nid = 0;
      if (ioctl(fd, NV_ESC_NUMA_INFO, &numa_info) == -1)
      {
          perror("ioctl get");
      }
      else
      {
        printf("%d\n", numa_info.nid);
        printf("%d\n", numa_info.status);
        printf("%llu\n", numa_info.memblock_size);
        printf("%llu\n", numa_info.numa_mem_addr);
        printf("%llu\n", numa_info.numa_mem_size);
        // nv_blacklist_addresses_t blacklist_addresses NV_ALIGN_BYTES(8);
      }
*/
/*
        nv_ioctl_card_info_t info;
        void *pinfo = &info;
        nv_ioctl_xfer_t xfer;
        xfer.cmd = NV_ESC_CARD_INFO;
        xfer.size = 65536;
        xfer.ptr = &pinfo;

        nv_ioctl_rm_api_version_t ver;

      if (ioctl(fd, NV_ESC_CHECK_VERSION_STR, &ver) == -1)
      {
          perror("ioctl get");
      }
      else
      {
        printf("%d\n", ver.reply);
      }
      
      exit(0);
*/
    int err = 0;
    int n;
    cudaGetDeviceCount(&n);

    for (int i = 0; i < n; ++i) {

        Table table;
        table.Header(0) = "Property";
        table.Header(1) = "Value";
        if (print_descriptions) {
                table.Header(2) = "Description";
        }


        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        table.Titlef("Device %d: %s", i, prop.name);

#if __CUDACC_VER_MAJOR__ > 8 && __CUDACC_VER_MINOR__ >  1
        table.NewRow();
        table.Cell("cudaDeviceProp.pageableMemoryAccessUsesHostPageTables");
        table.Cellf("%d", prop.pageableMemoryAccessUsesHostPageTables);
        if (print_descriptions) {
                table.Cell("Device accesses pageable memory using host page tables. This suggests Address Translation Services are enabled on Power9");
        }
#endif

#if __CUDACC_VER_MAJOR__ > 8
        table.NewRow();
        table.Cell("cudaDeviceProp.canUseHostPointerForRegisteredMem");
        table.Cellf("%d", prop.canUseHostPointerForRegisteredMem);
        if (print_descriptions) {
                table.Cell("Device can access host registered memory at the same virtual address as the CPU");
        }
#endif

#if __CUDACC_VER_MAJOR__ > 8 && __CUDACC_VER_MINOR__ >  1
{
        int v;
        cudaDeviceGetAttribute ( &v, cudaDevAttrDirectManagedMemAccessFromHost, i );
        table.NewRow();
        table.Cell("cudaDevAttrDirectManagedMemAccessFromHost");
        table.Cellf("%d", v);
        if (print_descriptions) {
                table.Cell("Host can directly access managed memory on the device without migration");
        }
}
#endif

#if __CUDACC_VER_MAJOR__ > 8 && __CUDACC_VER_MINOR__ >  1
{
        int v;
        cudaDeviceGetAttribute ( &v, cudaDevAttrCanFlushRemoteWrites, i );
        table.NewRow();
        table.Cell("cudaDevAttrCanFlushRemoteWrites");
        table.Cellf("%d", v);
        if (print_descriptions) {
                table.Cell("device supports flushing of outstanding remote writes");
        }
}
#endif

        table.NewRow();
        table.Cell("cudaDeviceProp.pageableMemoryAccess");
        table.Cellf("%d", prop.pageableMemoryAccess);
        if (print_descriptions) {
                table.Cell("Device supports coherently accessing pageable memory without calling cudaHostRegister on it.");
        }

        table.NewRow();
        table.Cell("cudaDeviceProp.concurrentManagedAccess");
        table.Cellf("%d", prop.concurrentManagedAccess);
        if (print_descriptions) {
                table.Cell("Device can coherently access managed memory concurrently with the CPU.");
        }

        table.NewRow();
        table.Cell("cudaDeviceProp.canMapHostMemory");
        table.Cellf("%d", prop.canMapHostMemory);
        if (print_descriptions) {
                table.Cell("Device can map host memory into the CUDA address space for use with cudaHostAlloc()/cudaHostGetDevicePointer()");
        }

        table.NewRow();
        table.Cell("cudaDeviceProp.totalGlobalMem");
        table.Cellf("%lu", prop.totalGlobalMem);
        if (print_descriptions) {
                table.Cell("bytes");
        }

        table.NewRow();
        table.Cell("cudaDeviceProp.totalConstMem");
        table.Cellf("%lu", prop.totalConstMem);
        if (print_descriptions) {
                table.Cell("bytes");
        }

        table.NewRow();
        table.Cell("cudaDeviceProp.sharedMemPerBlock");
        table.Cellf("%lu", prop.sharedMemPerBlock);
        if (print_descriptions) {
                table.Cell("bytes");
        }

        table.NewRow();
        table.Cell("cudaDeviceProp.sharedMemPerMultiprocessor");
        table.Cellf("%lu", prop.sharedMemPerMultiprocessor);
        if (print_descriptions) {
                table.Cell("bytes");
        }

        table.NewRow();
        table.Cell("cudaDeviceProp.l2CacheSize");
        table.Cellf("%d", prop.l2CacheSize);
        if (print_descriptions) {
                table.Cell("bytes");
        }

        table.NewRow();
        table.Cell("cudaDeviceProp.memoryBusWidth");
        table.Cellf("%d", prop.memoryBusWidth);
        if (print_descriptions) {
                table.Cell("bits");
        }

        table.NewRow();
        table.Cell("cudaDeviceProp.memoryClockRate");
        table.Cellf("%d", prop.memoryClockRate);
        if (print_descriptions) {
                table.Cell("kHz");
        }

        table.NewRow();
        table.Cell("cudaDeviceProp.asyncEngineCount");
        table.Cellf("%d", prop.asyncEngineCount);
        if (print_descriptions) {
                table.Cell("1: concurrent kernel and copy, 2: kernel and duplex copy");
        }

        table.NewRow();
        table.Cell("cudaDeviceProp.globalL1CacheSupported");
        table.Cellf("%d", prop.globalL1CacheSupported);
        if (print_descriptions) {
                table.Cell("Device supports caching of globals in L1 cache");
        }

        table.NewRow();
        table.Cell("cudaDeviceProp.localL1CacheSupported");
        table.Cellf("%d", prop.localL1CacheSupported);
        if (print_descriptions) {
                table.Cell("Device supports caching of locals in L1 cache");
        }

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

        if ("shell" == output_format) {
                printf("%s\n", report.ascii_str().c_str());
                printf("%s\n", table.shell_str().c_str());
        } else if ("csv" == output_format) {
                printf("%s\n", table.csv_str().c_str());
        } else if ("md" == output_format) {
                printf("%s\n", table.md_str().c_str());
        } else {
                fprintf(stderr, "unexpected output value: %s\n", output_format.c_str());
                return 1;
        }

    }



    return err;
}