#ifndef IGPU_H
#define IGPU_H

#include <cstddef>
#include <functional>
#include <span>
#include <unordered_map>

// TODO: Add more options as interface is built out
enum class GPUState { GPUSuccess, GPUFailure, GhostBuffer, InvalidDispatchType };

// Various states that can affect the level of optimization applied to memory buffers
// BufferUsage - How the buffer will be accessed during GPU tasks, configurable so buffers can be reused
// MemoryHint - How the data stored in the buffer will be treated throughout the lifetime of a task on a CPU/GPU level
//  - Enables optimizations with private memory on the GPU, guarantee that only it has access to it

// NOTE: Should BufferUsage be a more generalized data value? Might have some value on the CPU/other operations as well?
// Buffer Usage used for Scheduling purposes primarily (e.g. two kernels can read from a buffer at the same time)
enum class BufferUsage { ReadWrite, ReadOnly };
enum class MemoryHint { DeviceLocal, HostVisible };

class GPUBufferHandle {
  public:
    int ID;
    MemoryHint mem_hint;

    GPUBufferHandle() = default;
    GPUBufferHandle(int ID, MemoryHint mem_hint) : ID(ID), mem_hint(mem_hint) {};
    bool operator==(const GPUBufferHandle &other) const { return this->ID == other.ID; }
};

// GPUBufferHandle objects have effective hashes already since they store a unique ID
namespace std {
template <> struct std::hash<GPUBufferHandle> {
    std::size_t operator()(const GPUBufferHandle &buffer_handle) const noexcept {
        return std::hash<int>{}(buffer_handle.ID);
    }
};
} // namespace std

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

    bool operator==(const KernelDispatch &other) const { return this->kernel_name == other.kernel_name; };
};

namespace std {
template <> struct std::hash<KernelDispatch> {
    std::size_t operator()(const KernelDispatch &kernel_dispatch) const noexcept {
        return std::hash<std::string>{}(kernel_dispatch.kernel_name);
    }
};
} // namespace std

enum class DispatchType { Serial, Concurrent };

class IGPUExecutor {
  public:
    // Allocating/freeing buffer memory on the GPU for kernel tasks
    GPUBufferHandle virtual allocate_buffer(std::size_t buffer_size, const MemoryHint &mem_hint) = 0;
    GPUState virtual deallocate_buffer(const GPUBufferHandle &buffer_handle) = 0;

    // Sending memory between devices for task completion/after task completion
    GPUState virtual copy_to_device(std::span<const std::byte> data_mem, const GPUBufferHandle &buffer_handle,
                                    std::size_t data_size) = 0;
    GPUState virtual copy_from_device(std::span<std::byte> data_mem, const GPUBufferHandle &buffer_handle,
                                      std::size_t data_size, bool sync) = 0;

    GPUState virtual execute_batch(const std::vector<KernelDispatch> &kernels, const DispatchType &dispatch_type,
                                   std::function<void()> &cpu_callback) = 0;
    GPUState virtual execute_kernel(const KernelDispatch &kernel, std::function<void()> &cpu_callback) = 0;

    // Prevents more GPU tasks from being added until all current ones are complete
    GPUState virtual synchronize() = 0;

    // Allows for checking of buffer sizes without access to device specific buffer maps
    virtual int get_buffer_length(const GPUBufferHandle &buffer_handle) = 0;

    // Note this will default construct to false if value is not present - intended behavior here
    bool get_kernel_status(const std::string &kernel_name) { return kernel_status[kernel_name]; }

    virtual ~IGPUExecutor() = default;

  protected:
    // Allows for the scheduler to check the status of kernels it has dispatched (kernel name -> future promise)
    std::unordered_map<std::string, bool> kernel_status;
};

#endif
