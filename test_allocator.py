import torch
import time

my_alloc = torch.cuda.memory.CUDAPluggableAllocator('./alloc.so', 'my_malloc', 'my_free')
my_alloc_va = torch.cuda.memory.CUDAPluggableAllocator('./alloc.so', 'my_malloc_va', 'my_free_va')
my_alloc_vmm = torch.cuda.memory.CUDAPluggableAllocator('./alloc.so', 'vmm_alloc', 'vmm_free')

torch.cuda.memory.change_current_allocator(my_alloc_vmm)

size = 1024*1024;
while(True):
    print("torch.rand")
    b = torch.rand(size, device='cuda')
    print(f"tensor data_ptr: {hex(b.data_ptr())}")
    b.to("cpu")
    b = None
    #time.sleep(0.05)
    #print("torch.empty")
    b = torch.empty(size, device='cuda')
    print(f"tensor data_ptr: {hex(b.data_ptr())}")
    b = None
    
    print("torch.empty from cpu to cuda")
    b = torch.empty(size, device='cpu')
    b.to("cuda")
    print(f"tensor data_ptr: {hex(b.data_ptr())}")
    b = None
