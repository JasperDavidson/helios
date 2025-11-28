#ifndef IGPU_H
#define IGPU_H

#include "DataManager.h"
#include <cstddef>
#include <cstdint>
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
    std::vector<int> grid_dim;
    std::vector<int> block_dim;

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
    GPUBufferHandle virtual allocate_buffer(std::size_t buffer_size, const MemoryHint mem_hint) = 0;
    GPUState virtual deallocate_buffer(const GPUBufferHandle &buffer_handle) = 0;

    // Sending memory between devices for task completion/after task completion
    GPUState virtual copy_to_device(std::span<const std::byte> data_mem, const GPUBufferHandle &buffer_handle) = 0;
    GPUState virtual copy_from_device(std::span<std::byte> data_mem, const GPUBufferHandle &buffer_handle) = 0;

    GPUState virtual execute_batch(const std::vector<KernelDispatch> &kernels, const DispatchType &dispatch_type,
                                   std::function<void()> &cpu_callback) = 0;
    GPUState virtual execute_kernel(const KernelDispatch &kernel, std::function<void()> &cpu_callback) = 0;

    // Prevents more GPU tasks from being added until all current ones are complete
    GPUState virtual synchronize() = 0;

    // Note this will default construct to false if value is not present - intended behavior here
    bool get_kernel_status(const std::string &kernel_name) { return kernel_status_[kernel_name]; }

    void map_data_to_buffer(int data_id, GPUBufferHandle &buffer_handle) { data_buffer_map_[data_id] = buffer_handle; }
    GPUBufferHandle buffer_from_data(int data_id) { return data_buffer_map_[data_id]; };
    bool data_buffer_exists(int data_id) { return data_buffer_map_.find(data_id) != data_buffer_map_.end(); }

    virtual ~IGPUExecutor() = default;

  protected:
    // Class to handle memory allocation efficiently through the buddy system
    class GPUMemoryAllocator {
      public:
        GPUMemoryAllocator(size_t devloc_min_size, size_t devloc_max_size, size_t unified_min_size,
                           size_t unified_max_size, size_t hostvis_min_size, size_t hostvis_max_size);
        GPUMemoryAllocator();

        uint64_t next_pow2(uint64_t x) { return x <= 1 ? 1 : 1 << (64 - __builtin_clz(x - 1)); }
        size_t allocate_memory(size_t mem_size, MemoryHint mem_hint);
        void check_free_mem(size_t mem_size, size_t mem_offset, MemoryHint mem_hint);

        size_t devloc_min_order;
        size_t devloc_max_order;
        size_t hostvis_min_order;
        size_t hostvis_max_order;
        size_t unified_min_order;
        size_t unified_max_order;

        uint64_t devloc_free_mask = 0;
        uint64_t hostvis_free_mask = 0;
        uint64_t unified_free_mask = 0;

        std::unordered_map<uint8_t, std::vector<size_t>> devloc_size_address;
        std::unordered_map<uint8_t, std::vector<size_t>> unified_size_address;
        std::unordered_map<uint8_t, std::vector<size_t>> hostvis_size_address;

      private:
        std::unordered_map<MemoryHint, GPUBufferHandle> slab_map;

        std::unordered_map<size_t, std::unordered_map<size_t, size_t>> devloc_free_map;
        std::unordered_map<size_t, std::unordered_map<size_t, size_t>> unified_free_map;
        std::unordered_map<size_t, std::unordered_map<size_t, size_t>> hostvis_free_map;

        // helper function for selecting the current state variables based on memory type
        void init_mem_types(uint64_t *&free_mask, std::unordered_map<uint8_t, std::vector<size_t>> *&size_address,
                            std::unordered_map<size_t, std::unordered_map<size_t, size_t>> *&free_map,
                            MemoryHint mem_hint);
    };

    // Allows for the scheduler to check the status of kernels it has dispatched (kernel name -> future promise)
    // Is this needed still?
    std::unordered_map<std::string, bool> kernel_status_;

    // This mapping allows for checking which data has already been allocated to a buffer
    std::unordered_map<int, GPUBufferHandle> data_buffer_map_;
};

#endif
