#ifndef IGPU_H
#define IGPU_H

#include "DataManager.h"
#include <cstddef>
#include <functional>
#include <span>
#include <unordered_map>

// TODO: Add more options as interface is built out
enum class GPUState { GPUSuccess, GPUFailure, GhostBuffer, InvalidDispatchType };

// Encapsulate information about kernels
class KernelDispatch {
  public:
    std::string kernel_name;
    std::vector<GPUBufferHandle> buffer_handles;
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
                                    std::size_t data_size, bool sync) = 0;
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
    bool get_kernel_status(const std::string &kernel_name) { return kernel_status_[kernel_name]; }

    void map_data_to_buffer(int data_id, GPUBufferHandle &buffer_handle) { data_buffer_map_[data_id] = buffer_handle; }

    bool data_buffer_exists(int data_id) { return data_buffer_map_.find(data_id) != data_buffer_map_.end(); }

    virtual ~IGPUExecutor() = default;

  protected:
    // Allows for the scheduler to check the status of kernels it has dispatched (kernel name -> future promise)
    // Is this needed still?
    std::unordered_map<std::string, bool> kernel_status_;

    // This mapping allows for checking which data has already been allocated to a buffer
    std::unordered_map<int, GPUBufferHandle> data_buffer_map_;
};

#endif
