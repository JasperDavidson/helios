#ifndef METAL_H
#define METAL_H

#include "DataManager.h"
#include "IGPUExecutor.h"
#include <cstddef>
#include <memory>
#include <unordered_map>

class MetalExecutor : public IGPUExecutor {
  public:
    // If no device local (private) buffers will be used, no proxy buffer is needed
    MetalExecutor(std::pair<int, int> devloc_bounds, std::pair<int, int> hostvis_bounds,
                  std::pair<int, int> unified_bounds, size_t proxy_size = 0);
    ~MetalExecutor();

    GPUBufferHandle allocate_buffer(std::size_t buffer_size, const MemoryHint mem_hint) override;
    GPUState deallocate_buffer(const GPUBufferHandle &buffer_handle) override;

    GPUState copy_to_device(std::span<const std::byte> data_mem, const GPUBufferHandle &buffer_handle) override;
    GPUState copy_from_device(std::span<std::byte> data_mem, const GPUBufferHandle &buffer_handle) override;

    GPUState execute_batch(const std::vector<KernelDispatch> &kernels, const DispatchType &dispatch_type,
                           std::function<void()> &cpu_callback) override;
    GPUState execute_kernel(const KernelDispatch &kernel, std::function<void()> &cpu_callback) override;

    GPUState synchronize() override;

  private:
    int buffer_counter = 0;

    struct MetalExecutorImpl;
    std::unique_ptr<MetalExecutorImpl> p_metal_impl;

    // Reusable proxy buffer private resources
    GPUBufferHandle proxy_handle_;

    void load_default_library();

    // Helper functions for managing memory transfers
    GPUState blit_to_private(std::span<const std::byte> data_mem, const GPUBufferHandle &buffer_handle);
    GPUState copy_to_shared(std::span<const std::byte> data_mem, const GPUBufferHandle &buffer_handle);
    GPUState copy_to_managed(std::span<const std::byte> data_mem, const GPUBufferHandle &buffer_handle);
    GPUState private_to_cpu(std::span<std::byte> data_mem, const GPUBufferHandle &buffer_handle);
    GPUState managed_to_cpu(std::span<std::byte> data_mem, const GPUBufferHandle &buffer_handle);
    GPUState shared_to_cpu(std::span<std::byte> data_mem, const GPUBufferHandle &buffer_handle);

    GPUMemoryAllocator mem_allocator;

    // Provides access to proxy buffer and extends size if needed
    void access_proxy(size_t data_size);
};

#endif
