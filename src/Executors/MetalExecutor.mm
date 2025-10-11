#include "MetalExecutor.h"
#include "Metal/Metal.h"
#include <Foundation/Foundation.h>
#include <Foundation/NSObjCRuntime.h>
#include <Metal/Metal.h>
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
  cmd_queue_ = [mtl_device_ newCommandQueue];
  load_default_library();

  kernel_map_ = std::unordered_map<std::string, id<MTLComputePipelineState>>();
};

MetalExecutor::MetalExecutor() : mtl_device_(MTLCreateSystemDefaultDevice()) {
  initialize();
};
