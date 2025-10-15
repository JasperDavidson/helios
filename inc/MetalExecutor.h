#ifndef METAL_H
#define METAL_H

#include "IGPUExecutor.h"
#include <Foundation/Foundation.h>
#include <Foundation/NSObjCRuntime.h>
#include <cstdint>
#include <unordered_map>

@protocol MTLDevice;
@protocol MTLCommandQueue;
@protocol MTLBuffer;
@protocol MTLLibrary;
@protocol MTLComputePipelineState;

class MetalExecutor : public IGPUExecutor {
  public:
    MetalExecutor();

    GPUBufferHandle allocate_buffer(std::uint32_t buffer_size, const BufferUsage &buffer_usage,
                                    const MemoryHint &mem_hint) override;
    GPUState deallocate_buffer(const GPUBufferHandle &buffer_handle) override;

    GPUState copy_to_device(std::span<const std::byte> data_mem, const GPUBufferHandle &buffer_handle,
                            std::uint32_t data_size) override;
    GPUState copy_from_device(std::span<std::byte> data_mem, const GPUBufferHandle &buffer_handle,
                              std::uint32_t data_size) override;

    GPUState execute_batch(const std::vector<KernelDispatch> &kernels, const DispatchType &dispatch_type) override;
    GPUState execute_kernel(const KernelDispatch &kernel) override;

    GPUState synchronize() override;

  private:
    static constexpr NSString *const LIBRARY_NAME = @"kernels";

    // Metal specific variables
    id<MTLDevice> mtl_device_;
    id<MTLCommandQueue> command_queue_;
    id<MTLLibrary> mtl_library_;

    // Hashmap for storing already prepared compute pipelines
    std::unordered_map<std::string, id<MTLComputePipelineState>> pipeline_map_;
    // Hashmap for storing GPU buffer handles
    int buffer_counter = 0;
    std::unordered_map<GPUBufferHandle, id<MTLBuffer>> buffer_map_;

    void initialize();
    void load_default_library();
    id<MTLComputePipelineState> find_cache_pipeline(const std::string &kernel_name);
};

#endif
