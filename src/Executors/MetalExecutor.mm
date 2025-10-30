#include "MetalExecutor.h"
#include "IGPUExecutor.h"
#include "Metal/Metal.h"
#include <Foundation/Foundation.h>
#include <Foundation/NSObjCRuntime.h>
#include <MacTypes.h>
#include <Metal/Metal.h>
#include <_string.h>
#include <algorithm>
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
}

id<MTLComputePipelineState> MetalExecutor::find_cache_pipeline(const std::string &kernel_name) {
    auto cache_iter = pipeline_map_.find(kernel_name);

    if (cache_iter != pipeline_map_.end()) {
        return cache_iter->second;
    }

    NSString *ns_name = [NSString stringWithUTF8String:kernel_name.c_str()];
    id<MTLFunction> kernel_func = [mtl_library_ newFunctionWithName:ns_name];
    if (kernel_func == nil) {
        throw std::runtime_error("No kernel function with name: " + kernel_name + " was found");
    }

    NSError *pipeline_creation_error;
    id<MTLComputePipelineState> compute_pipeline =
        [mtl_device_ newComputePipelineStateWithFunction:kernel_func error:&pipeline_creation_error];
    if (compute_pipeline == nil) {
        std::string error_message = "Failed to create compute pipeline for the kernel function: " + kernel_name;
        error_message += [[pipeline_creation_error localizedDescription] UTF8String];
        throw std::runtime_error(error_message);
    }

    pipeline_map_[kernel_name] = compute_pipeline;
    return compute_pipeline;
}

MetalExecutor::MetalExecutor() : mtl_device_(MTLCreateSystemDefaultDevice()) {
    if (!mtl_device_) {
        throw std::runtime_error("Failed to get Metal device.");
    }
    command_queue_ = [mtl_device_ newCommandQueue];
    load_default_library();

    pipeline_map_ = std::unordered_map<std::string, id<MTLComputePipelineState>>();
}

GPUBufferHandle MetalExecutor::allocate_buffer(std::uint32_t buffer_size, const MemoryHint &mem_hint) {
    // Create the buffer handle object
    GPUBufferHandle buffer_handle(buffer_counter, mem_hint);
    buffer_counter++;

    // Create the buffer and map to its handle options:0 -> default storage option for the system (or private)
    id<MTLBuffer> buffer;
    if (mem_hint == MemoryHint::DeviceLocal) {
        buffer = [mtl_device_ newBufferWithLength:buffer_size options:MTLResourceStorageModePrivate];
    } else if (mem_hint == MemoryHint::HostVisible) {
        buffer = [mtl_device_ newBufferWithLength:buffer_size options:0];
    }
    buffer_map_[buffer_handle] = buffer;

    return buffer_handle;
}

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

        // Placed in same queue as compute encoder so no need to block
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

    // For private resources: Create a proxy buffer that can contain private data and transfer to CPU
    // For shared resources: Queue an empty command buffer --> "finishes" after GPU write is complete
    // For managed resources: CPU/GPU have unique copies - block to ensure data transfer is complete
    if (([buffer resourceOptions] & MTLResourceStorageModePrivate) == MTLResourceStorageModePrivate) {
        // Use a proxy buffer to transfer data
        id<MTLBuffer> proxy_buffer = [mtl_device_ newBufferWithLength:data_size options:0];
        id<MTLCommandBuffer> transfer_cmd_buffer = [command_queue_ commandBuffer];
        id<MTLBlitCommandEncoder> blit_encoder = [transfer_cmd_buffer blitCommandEncoder];

        [blit_encoder copyFromBuffer:buffer sourceOffset:0 toBuffer:proxy_buffer destinationOffset:0 size:data_size];
        [blit_encoder endEncoding];
        [transfer_cmd_buffer commit];
        [transfer_cmd_buffer waitUntilCompleted];

        void *proxy_mem = [proxy_buffer contents];
        memcpy(data_mem.data(), proxy_mem, data_size);
    } else if (([buffer resourceOptions] & MTLResourceStorageModeManaged) == MTLResourceStorageModeManaged) {
        id<MTLCommandBuffer> synchronize_cmd_buffer = [command_queue_ commandBuffer];
        id<MTLBlitCommandEncoder> blit_encoder = [synchronize_cmd_buffer blitCommandEncoder];

        [blit_encoder synchronizeResource:buffer];
        [blit_encoder endEncoding];
        [synchronize_cmd_buffer commit];
        [synchronize_cmd_buffer waitUntilCompleted];

        void *buffer_mem = [buffer contents];
        memcpy(data_mem.data(), buffer_mem, data_size);
    } else if (([buffer resourceOptions] & MTLResourceStorageModeShared) == MTLResourceStorageModeShared) {
        // Inefficient synchronize() call? Needed to be *safe* since it ensures data is fully written to before reading
        // Task scheduler might be able to optimize task placement such that this isn't a big deal
        synchronize();

        void *buffer_mem = [buffer contents];
        memcpy(data_mem.data(), buffer_mem, data_size);
    }

    // Invalid buffer storage type - shouldn't trigger
    return GPUState::GPUFailure;
}

GPUState MetalExecutor::execute_batch(const std::vector<KernelDispatch> &kernels, const DispatchType &dispatch_type) {
    for (const KernelDispatch &kernel : kernels) {
        // Construct a buffer for each kernel, needed to allow for end commands for individual kernel completion
        id<MTLCommandBuffer> compute_buffer = [command_queue_ commandBuffer];

        // Prepare the compute encoder (binding GPU resources, providing compute pipeline, etc.)
        id<MTLComputeCommandEncoder> compute_encoder;
        if (dispatch_type == DispatchType::Serial) {
            compute_encoder = [compute_buffer computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
        } else if (dispatch_type == DispatchType::Concurrent) {
            compute_encoder = [compute_buffer computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        } else {
            return GPUState::InvalidDispatchType;
        }

        // Fetch the compute pipeline or create it if needed
        id<MTLComputePipelineState> compute_pipeline = find_cache_pipeline(kernel.kernel_name);

        [compute_encoder setComputePipelineState:compute_pipeline];

        for (int j = 0; j < kernel.buffer_bindings.size(); ++j) {
            // TODO: Handle if buffer is not in the map (invalid buffer, NEVER ALLOCATED)
            id<MTLBuffer> bind_buffer = buffer_map_[kernel.buffer_bindings[j].buffer_handle];
            BufferUsage buffer_usage = kernel.buffer_bindings[j].buffer_usage;
            MTLResourceUsage mtl_usage =
                (buffer_usage == BufferUsage::ReadWrite) ? MTLResourceUsageWrite : MTLResourceUsageRead;

            // Each buffer must be bound at a unique index
            // Usage can be encoded based on how the kernel plans to use them --> potential reuse by future kernels
            [compute_encoder useResource:bind_buffer usage:mtl_usage];
            [compute_encoder setBuffer:bind_buffer offset:0 atIndex:j];
        }

        MTLSize groups_per_grid = MTLSizeMake(kernel.kernel_size[0], kernel.kernel_size[1], kernel.kernel_size[2]);
        MTLSize threads_per_group =
            MTLSizeMake(kernel.threads_per_group[0], kernel.threads_per_group[1], kernel.threads_per_group[2]);
        [compute_encoder dispatchThreadgroups:groups_per_grid threadsPerThreadgroup:threads_per_group];

        // Update when each kernel ends individually, rather than when the entire batch ends
        [compute_buffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
          kernel_status[kernel.kernel_name] = true;
        }];

        [compute_encoder endEncoding];
        [compute_buffer commit];
    }

    return GPUState::GPUSuccess;
}

GPUState MetalExecutor::execute_kernel(const KernelDispatch &kernel) {
    return execute_batch({kernel}, DispatchType::Serial);
}

// Add a final empty command buffer and block until it finishes
GPUState MetalExecutor::synchronize() {
    id<MTLCommandBuffer> synchronize_buffer = [command_queue_ commandBuffer];

    [synchronize_buffer commit];
    [synchronize_buffer waitUntilCompleted];

    return GPUState::GPUSuccess;
}

int MetalExecutor::get_buffer_length(const GPUBufferHandle &buffer_handle) {
    return [buffer_map_[buffer_handle] length];
}
