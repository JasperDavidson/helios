#ifndef IGPU_H
#define IGPU_H

#include <span>

// TODO: Add more options as interface is built out
enum class GPUState { GPUSuccess, GPUFailure, GhostBuffer, InvalidDispatchType };

// Various states that can affect the level of optimization applied to memory buffers
// BufferUsage - How the buffer will be accessed in the global system
enum class BufferUsage { ReadWrite, ReadOnly };
enum class MemoryHint { DeviceLocal, HostVisible };

class GPUBufferHandle {
  public:
    int ID;
    MemoryHint mem_hint;

    GPUBufferHandle(int ID, MemoryHint mem_hint) : ID(ID), mem_hint(mem_hint) {};
};

// Encapsulate information about buffers and kernels

// Separate so that buffer handles can be reused across kernels
struct BufferBinding {
    GPUBufferHandle buffer_handle;
    BufferUsage buffer_usage;
};

class KernelDispatch {
  public:
    std::string kernel_name;
    std::vector<BufferBinding> buffer_bindings;
    std::vector<int> kernel_size;
    std::vector<int> threads_per_group;
};

enum class DispatchType { Serial, Concurrent };

class IGPUExecutor {
  public:
    // Allocating/freeing buffer memory on the GPU for kernel tasks
    GPUBufferHandle virtual allocate_buffer(std::uint32_t buffer_size, const MemoryHint &mem_hint) = 0;
    GPUState virtual deallocate_buffer(const GPUBufferHandle &buffer_handle) = 0;

    // Sending memory between devices for task completion/after task completion
    GPUState virtual copy_to_device(std::span<const std::byte> data_mem, const GPUBufferHandle &buffer_handle,
                                    std::uint32_t data_size) = 0;
    GPUState virtual copy_from_device(std::span<std::byte> data_mem, const GPUBufferHandle &buffer_handle,
                                      std::uint32_t data_size) = 0;

    GPUState virtual execute_batch(const std::vector<KernelDispatch> &kernels, const DispatchType &dispatch_type) = 0;
    GPUState virtual execute_kernel(const KernelDispatch &kernel) = 0;

    // Prevents more GPU tasks from being added until all current ones are
    // complete
    GPUState virtual synchronize() = 0;

    // Allows for checking of buffer sizes without access to device specific buffer maps
    virtual int get_buffer_length(const GPUBufferHandle &buffer_handle) = 0;

    virtual ~IGPUExecutor() = default;
};

#endif
