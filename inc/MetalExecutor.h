#ifndef METAL_H
#define METAL_H

#include "IGPUExecutor.h"
#include <Foundation/Foundation.h>
#include <Foundation/NSObjCRuntime.h>
#include <cstdint>
#include <unordered_map>

@protocol MTLDevice;
@protocol MTLCommandQueue;
@protocol MTLLibrary;
@protocol MTLComputePipelineState;

id<MTLLibrary> fetch_library(id<MTLDevice> device, NSString *library_name);

class MetalExecutor : public IGPUExecutor {
public:
  MetalExecutor();

  GPUBufferHandle allocate_buffer(std::uint32_t buffer_size) override;
  GPUState deallocate_buffer(const GPUBufferHandle &buffer_handle) override;

  GPUState copy_to_device(std::span<const std::byte> data_mem,
                          const GPUBufferHandle &buffer_handle,
                          std::uint32_t data_size) override;
  GPUState copy_from_device(std::span<std::byte> data_mem,
                            const GPUBufferHandle &buffer_handle,
                            std::uint32_t data_size) override;

  GPUState execute_kernel(const std::string &kernel_name,
                          const std::vector<GPUBufferHandle> &buffer_handles,
                          const std::vector<int> &kernel_size) override;

  GPUState synchronize() override;

private:
  static constexpr NSString *const LIBRARY_NAME = @"kernels";

  // Metal specific variables
  id<MTLDevice> mtl_device_;
  id<MTLCommandQueue> cmd_queue_;
  id<MTLLibrary> mtl_library_;

  // Hashmap for storing already prepared compute pipelines
  std::unordered_map<std::string, id<MTLComputePipelineState>> kernel_map_;

  void initialize();
  void load_default_library();
};

#endif
