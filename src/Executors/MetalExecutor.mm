#include "MetalExecutor.h"
#include "IGPUExecutor.h"
#include "Metal/Metal.h"
#include <Foundation/Foundation.h>
#include <Foundation/NSObjCRuntime.h>
#include <Metal/Metal.h>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <objc/NSObjCRuntime.h>
#include <stdexcept>
#include <unordered_map>

void MetalExecutor::load_default_library() {
    NSURL *library_url = [[NSBundle mainBundle] URLForResource:LIBRARY_NAME withExtension:@"metallib"];

    if (!library_url) {
        throw std::runtime_error("Failed to find the default kernel library url.");
    }

    NSError *library_error = nil;
    id<MTLLibrary> library = [mtl_device_ newLibraryWithURL:library_url error:&library_error];

    if (library == nil) {
        std::string error_message = "Failed to fetch default kernel library. Description: ";
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

MetalExecutor::MetalExecutor() : mtl_device_(MTLCreateSystemDefaultDevice()) { initialize(); };

GPUBufferHandle MetalExecutor::allocate_buffer(std::uint32_t buffer_size, const BufferUsage &buffer_usage,
                                               const MemoryHint &mem_hint) {
    // Create the buffer handle object
    GPUBufferHandle buffer_handle(buffer_counter, buffer_usage, mem_hint);
    buffer_counter++;

    // Create the buffer and map to its handle options:0 -> default storage option for the system (or private)
    id<MTLBuffer> buffer = [mtl_device_ newBufferWithLength:buffer_size options:0];
    buffer_map_[buffer_handle] = buffer;

    return buffer_handle;
};

GPUState MetalExecutor::deallocate_buffer(const GPUBufferHandle &buffer_handle) {
    // Remove the reference to the command buffer - Metal will automatically deallocate resources
    if (buffer_map_.contains(buffer_handle)) {
        buffer_map_.erase(buffer_handle);

        return GPUState::GPUSuccess;
    }

    return GPUState::GhostBuffer;
}

// TODO: Perhaps we want to have some sort of MemoryAllocator that can
// determine when it's best to use multiple buffers vs. one large one?
GPUState MetalExecutor::copy_to_device(std::span<const std::byte> data_mem, const GPUBufferHandle &buffer_handle,
                                       std::uint32_t data_size) {
    auto buffer_it = buffer_map_.find(buffer_handle);
    if (buffer_it == buffer_map_.end()) {
        return GPUState::GhostBuffer;
    }

    id<MTLBuffer> buffer = buffer_it->second;

    // For private resources: Create a proxy buffer that can transfer to private
    // For shared resources: Compute pipeline can ensure synchronization
    // For managed resources: Manually synchronized, CPU/GPU have unique copies
    if (([buffer resourceOptions] & MTLResourceStorageModePrivate) == MTLResourceStorageModePrivate) {
        id<MTLBuffer> proxy_buffer = [mtl_device_ newBufferWithBytes:data_mem.data() length:data_size options:0];
        id<MTLCommandBuffer> transfer_cmd_buffer = [command_queue_ commandBuffer];
        id<MTLBlitCommandEncoder> blit_encoder = [transfer_cmd_buffer blitCommandEncoder];

        [blit_encoder copyFromBuffer:proxy_buffer sourceOffset:0 toBuffer:buffer destinationOffset:0 size:data_size];
        [blit_encoder endEncoding];
        [transfer_cmd_buffer commit];

        return GPUState::GPUSuccess;
    } else if (([buffer resourceOptions] & MTLResourceStorageModeManaged) == MTLResourceStorageModeManaged) {
        void *buffer_mem = [buffer contents];
        memcpy(buffer_mem, data_mem.data(), data_size);
        [buffer didModifyRange:NSMakeRange(0, data_size)];

        return GPUState::GPUSuccess;
    } else if (([buffer resourceOptions] & MTLResourceStorageModeShared) == MTLResourceStorageModeShared) {
        void *buffer_mem = [buffer contents];
        memcpy(buffer_mem, data_mem.data(), data_size);

        return GPUState::GPUSuccess;
    }

    // Invalid buffer storage type - shouldn't trigger
    return GPUState::GPUFailure;
}

GPUState MetalExecutor::copy_from_device(std::span<std::byte> data_mem, const GPUBufferHandle &buffer_handle,
                                         std::uint32_t data_size) {
    auto buffer_it = buffer_map_.find(buffer_handle);
    if (buffer_it == buffer_map_.end()) {
        return GPUState::GhostBuffer;
    }

    id<MTLBuffer> buffer = buffer_it->second;

    if (([buffer resourceOptions] & MTLResourceStorageModePrivate) == MTLResourceStorageModePrivate) {

    } else if (([buffer resourceOptions] & MTLResourceStorageModeManaged) == MTLResourceStorageModeManaged) {
        id<MTLCommandBuffer> synchronize_cmd_buffer = [command_queue_ commandBuffer];
        id<MTLBlitCommandEncoder> blit_encoder = [synchronize_cmd_buffer blitCommandEncoder];

        [blit_encoder synchronizeResource:buffer];
        [blit_encoder endEncoding];
        [synchronize_cmd_buffer commit];

        // Need to wait until data is actually sent over
        // Need to copy data from the buffer now
    } else if (([buffer resourceOptions] & MTLResourceStorageModeShared) == MTLResourceStorageModeShared) {
    }

    return GPUState::GPUFailure;
}
