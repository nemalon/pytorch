#include <cuda_runtime_api.h>
#include <cuda.h>
#include <iostream>
#include <assert.h>
extern "C" {

    // Define the CUDA error-checking macro
    #define CHECK_CUDA(ans) { gpuAssert((ans), __FILE__, __LINE__); }

    // Helper function to handle CUDA errors
    inline void gpuAssert(CUresult result, const char* file, int line, bool abort = true) {
        if (result!= CUDA_SUCCESS) {
            const char* error_string;
            cuGetErrorString(result, &error_string);
            fprintf(stderr, "CUDA Error: %s (%s:%d)\n", error_string, file, line);
            if (abort) {
                exit(EXIT_FAILURE);
            }
        }
    }
    CUmemAllocationProp g_prop = {};
    size_t g_granularity;    
    size_t g_page_size = 4*1024*1024LLU;
    __attribute__((constructor))
    static void lib_init() {
        CUresult result;
    	int supportsVMM = 0;
        /* TODO: we might want to change that */
        CUdevice device_id = 0;
	    CUcontext ctx;
	    unsigned int ctx_create_flags = 0;

        printf("Init allocator lib\n");
        unsigned int init_flags = 0;
        result = cuInit(init_flags);
	    result = cuCtxCreate(&ctx, ctx_create_flags, device_id);
    	result = cuDeviceGetAttribute(&supportsVMM, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device_id);
    	fprintf(stderr, "SupportsVMM: %d\n", supportsVMM);
    	// Note: The created context is also made the current context,
    	// so we are _in_ a context from now on.

        g_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
	    g_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        
	    g_prop.location.id = (int) 0;
	    g_prop.win32HandleMetaData = NULL;

        result = cuMemGetAllocationGranularity (&g_granularity, &g_prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM );
        printf("minimum granularity %lu\n", g_granularity);
    }
    CUdeviceptr _malloc_single_va(size_t size, int device, CUdeviceptr requested) {
        CUmemGenericAllocationHandle handle;
        CUresult result;
	    CUdeviceptr ptr;

	    size_t padded_size = ((g_granularity + size - 1) / g_granularity) * g_granularity;
   		printf("Reserve a virtual address range %lu (%lu)\n", size, padded_size);
		result = cuMemAddressReserve(&ptr, padded_size, 0 /*alignment*/, requested, 0);
        CHECK_CUDA(result);
		printf("ptr = 0x%llx requested = 0x%llx\n", ptr, requested);
		printf("create a chunk of memory %lu\n", padded_size);
		result = cuMemCreate(&handle, padded_size, &g_prop, 0);
        CHECK_CUDA(result);
		printf("Map the virtual address range to the physical allocation %lu\n", padded_size);
		result = cuMemMap(ptr, padded_size, 0, handle, 0); 
        CHECK_CUDA(result);
        /* we can change the access for all of the vas
        printf("Set access %lu\n", padded_size);
        CUmemAccessDesc accessDesc;
	    accessDesc.location = g_prop.location;
	    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
	    result = cuMemSetAccess(ptr, padded_size, &accessDesc, 1ULL);
        */

        return  ptr;
    }
    void *my_malloc_va(size_t size, int device, cudaStream_t stream) {
        CUmemGenericAllocationHandle handle;
        CUresult result;
	    CUdeviceptr requested = 0;
	    CUdeviceptr ptr;
        CUdeviceptr base;

	    size_t padded_size = ((g_granularity + size - 1) / g_granularity) * g_granularity;
        if (padded_size < g_page_size)
            padded_size = g_page_size; 
        size_t pages = padded_size/g_page_size ;
        printf("alloc size=%lu padded_size=%lu pages=%lu\n", size, padded_size, pages);
        base = ptr = _malloc_single_va(g_page_size, device, requested);
        for (size_t page = 1; page < pages; page++) {
            requested = ptr + g_page_size;
            ptr = _malloc_single_va(g_page_size, device, requested);
            assert((void*) ptr != nullptr);
        }
        printf("Set access %lu\n", padded_size);
        CUmemAccessDesc accessDesc;
	    accessDesc.location = g_prop.location;
	    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
	    result = cuMemSetAccess(base, padded_size, &accessDesc, 1ULL);
        CHECK_CUDA(result);
        return (void*) base;
    }

    void my_free_va(void* ptr, size_t size, int device, cudaStream_t stream){
        CUresult result;
        std::cout << "allocator free " << ptr << " " << stream << std::endl;
	    size_t padded_size = ((g_granularity + size - 1) / g_granularity) * g_granularity;
        CUdeviceptr p = (CUdeviceptr) ptr;
        size_t pages = padded_size/g_page_size;
        for (size_t page = 0; page < pages; ++page) {
            printf("GetAllocationHandle %lu (%lu)\n", size, g_page_size);
            CUmemGenericAllocationHandle handle;
            result = cuMemRetainAllocationHandle (&handle , (void*)p );
            CHECK_CUDA(result);
       	    printf("Unmap %lu (%lu)\n", size, g_page_size);
            result = cuMemUnmap(p, g_page_size);
            CHECK_CUDA(result);
            printf("Release handle %lu (%lu)\n", size, g_page_size);
            result = cuMemRelease(handle);
            CHECK_CUDA(result);
           	printf("Release virtual address range %lu (%lu)\n", size, g_page_size);
            result = cuMemAddressFree(p, g_page_size);
            CHECK_CUDA(result);
            p += g_page_size;
        }
    }

void* vmm_alloc(size_t size, int device, cudaStream_t stream) {
        std::cout << "allocator alloc " << size << " " << stream << std::endl;
        CUmemAllocationProp prop = {};
        prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id   = device;

        size_t granularity = g_granularity;

        size_t original_size = size;
        size = ((size - 1) / granularity + 1) * granularity;
        std::cout << "allocator alloc " << size << " " << original_size <<" " << stream << std::endl;
        CUdeviceptr dptr;
        if (cuMemAddressReserve(&dptr, size, 0, 0, 0) != CUDA_SUCCESS) {
            return NULL;
        }
        for (size_t offset = 0 ; offset < size; offset += granularity)
        {
            std::cout << "allocate chunk at offset " << offset << std::endl;
            /* alloc handles */
            CUmemGenericAllocationHandle allocationHandle;
            if (cuMemCreate(&allocationHandle, granularity, &prop, 0) != CUDA_SUCCESS) {
                return NULL;
            }

            if (cuMemMap(dptr+offset, granularity, 0, allocationHandle, 0) != CUDA_SUCCESS) {
                return NULL;
            }
            /* this release the handle, 
            the backing memory will be freed only when the Mapping is release */
            if (cuMemRelease(allocationHandle) != CUDA_SUCCESS) {
                return NULL;
            }
        }

        CUmemAccessDesc accessDescriptor;
        accessDescriptor.location.id   = prop.location.id;
        accessDescriptor.location.type = prop.location.type;
        accessDescriptor.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        if (cuMemSetAccess(dptr, size, &accessDescriptor, 1) != CUDA_SUCCESS) {
            return NULL;
        }

        void *ptr = (void *)dptr;

        return ptr;
    }

    void vmm_free(void* ptr, size_t size, int device, cudaStream_t stream){ 

        CUresult result;
        if (!ptr)
            return ;

        size_t granularity = g_granularity;

        size = ((size - 1) / granularity + 1) * granularity;
        std::cout << "allocator free " << size << " " << stream << std::endl;        
/*        printf("GetAllocationHandle %lu (%lu)\n", size, g_page_size);
        CUmemGenericAllocationHandle handle;
        result = cuMemRetainAllocationHandle (&handle , ptr);
        CHECK_CUDA(result);
        result = cuMemRelease(handle);
*/
        if (cuMemUnmap((CUdeviceptr)ptr, size) != CUDA_SUCCESS ||
            cuMemAddressFree((CUdeviceptr)ptr, size) != CUDA_SUCCESS) {
            return;
        }
    }

    void *my_malloc(size_t size, int device, cudaStream_t stream) {
        void *ptr = NULL;
        cudaError_t err = cudaMalloc (&ptr,size);
        std::cout << "allocator alloc " << ptr << " " <<size << "" << err<<std::endl;
        return ptr;
    }

    void my_free(void* ptr, size_t size, int device, cudaStream_t stream){
        std::cout << "allocator free " << ptr << " " << stream << std::endl;
        cudaFree(ptr);
    }
}
