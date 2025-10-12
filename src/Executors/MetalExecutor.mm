#include "MetalExecutor.h"
#include "IGPUExecutor.h"
#include "Metal/Metal.h"
#include <Foundation/Foundation.h>
#include <Foundation/NSObjCRuntime.h>
#include <Metal/Metal.h>
#include <cstdint>
#include <objc/NSObjCRuntime.h>
#include <stdexcept>
#include <unordered_map>

void MetalExecutor::load_default_library() {
  NSURL *library_url = [[NSBundle mainBundle] URLForResource:LIBRARY_NAME
                                               withExtension:@"metallib"];

  if (!library_url) {
    throw std::runtime_error("Failed to find the default kernel library url.");
  }

  NSError *library_error = nil;
  id<MTLLibrary> library = [mtl_device_ newLibraryWithURL:library_url
                                                    error:&library_error];

  if (library == nil) {
    std::string error_message =
        "Failed to fetch default kernel library. Description: ";
    error_message += [[library_error localizedDescription] UTF8String];
    throw std::runtime_error(error_message);
  }

  mtl_library_ = library;
};

void MetalExecutor::initialize() {
  if (!mtl_device_) {
    throw std::runtime_error("Failed to get Metal device.");
  }
  command_queue_ = [mtl_device_ newCommandQueue];
  load_default_library();

  kernel_map_ = std::unordered_map<std::string, id<MTLComputePipelineState>>();
};

MetalExecutor::MetalExecutor() : mtl_device_(MTLCreateSystemDefaultDevice()) {
  initialize();
};

GPUBufferHandle MetalExecutor::allocate_buffer(std::uint32_t buffer_size) {
  // Create the buffer handle object
  GPUBufferHandle buffer_handle(buffer_counter);
  buffer_counter++;

  // Create the buffer and map to its handle
  // options:0 -> default storage option for the system
  id<MTLBuffer> buffer = [mtl_device_ newBufferWithLength:buffer_size
                                                  options:0];
  buffer_map_[buffer_handle] = buffer;

  return buffer_handle;
};

GPUState
MetalExecutor::deallocate_buffer(const GPUBufferHandle &buffer_handle) {
  // Remove the reference to the command buffer - Metal will automatically
  // deallocate resources
  if (buffer_map_.contains(buffer_handle)) {
    buffer_map_.erase(buffer_handle);

    return GPUState::GPUSuccess;
  }

  return GPUState::GhostBuffer;
}
