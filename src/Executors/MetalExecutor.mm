#include "MetalExecutor.h"
#include "IGPUExecutor.h"
#include "Metal/Metal.h"
#include <Foundation/Foundation.h>
#include <Foundation/NSObjCRuntime.h>
#include <MacTypes.h>
#include <Metal/Metal.h>
#include <Security/cssmconfig.h>
#include <_string.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <future>
#include <iterator>
#include <objc/NSObjCRuntime.h>
#include <stdexcept>
#include <unordered_map>

static constexpr NSString *const LIBRARY_NAME = @"kernels";

struct MetalExecutor::MetalExecutorImpl {
    // Metal specific variables
    id<MTLDevice> mtl_device_;
    id<MTLCommandQueue> command_queue_;
    id<MTLLibrary> mtl_library_;

    // map for storing already prepared compute pipelines
    std::unordered_map<std::string, id<MTLComputePipelineState>> pipeline_map_;

    // map for storing GPU buffer handles
    std::unordered_map<GPUBufferHandle, id<MTLBuffer>> buffer_map_;

    id<MTLComputePipelineState> find_cache_pipeline(const std::string &kernel_name) {
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
};

void MetalExecutor::load_default_library() {
    NSURL *library_url = [[NSBundle mainBundle] URLForResource:LIBRARY_NAME withExtension:@"metallib"];

    if (!library_url) {
        throw std::runtime_error("Failed to find the default kernel library url.");
    }

    NSError *library_error = nil;
    id<MTLLibrary> library = [p_metal_impl->mtl_device_ newLibraryWithURL:library_url error:&library_error];

    if (library == nil) {
        std::string error_message = "Failed to fetch default kernel library. Description: ";
        error_message += [[library_error localizedDescription] UTF8String];
        throw std::runtime_error(error_message);
    }

    p_metal_impl->mtl_library_ = library;
}

MetalExecutor::MetalExecutor(size_t proxy_size) : p_metal_impl(std::make_unique<MetalExecutorImpl>()) {
    p_metal_impl->mtl_device_ = MTLCreateSystemDefaultDevice();
    if (!p_metal_impl->mtl_device_) {
        throw std::runtime_error("Failed to get Metal device.");
    }
    p_metal_impl->command_queue_ = [p_metal_impl->mtl_device_ newCommandQueue];
    load_default_library();

    p_metal_impl->pipeline_map_ = std::unordered_map<std::string, id<MTLComputePipelineState>>();
    proxy_buffer_ = allocate_buffer(proxy_size, MemoryHint::DeviceLocal);
}

MetalExecutor::~MetalExecutor() = default;

GPUBufferHandle MetalExecutor::allocate_buffer(std::size_t buffer_size, const MemoryHint &mem_hint) {
    // Create the buffer handle object
    GPUBufferHandle buffer_handle(buffer_counter, mem_hint);
    buffer_counter++;

    // Create the buffer and map to either private or managed storage load
    // NOTE: Shared resources are not as useful due to needing to manage synchronization to avoid data races
    id<MTLBuffer> buffer;
    if (mem_hint == MemoryHint::DeviceLocal) {
        buffer = [p_metal_impl->mtl_device_ newBufferWithLength:buffer_size options:MTLResourceStorageModePrivate];
    } else if (mem_hint == MemoryHint::HostVisible) {
        buffer = [p_metal_impl->mtl_device_ newBufferWithLength:buffer_size options:MTLResourceStorageModeManaged];
    }
    p_metal_impl->buffer_map_[buffer_handle] = buffer;

    return buffer_handle;
}

GPUState MetalExecutor::deallocate_buffer(const GPUBufferHandle &buffer_handle) {
    // Remove the reference to the command buffer - Metal will automatically deallocate resources
    if (p_metal_impl->buffer_map_.contains(buffer_handle)) {
        p_metal_impl->buffer_map_.erase(buffer_handle);

        return GPUState::GPUSuccess;
    }

    return GPUState::GhostBuffer;
}

// NOTE: Should implement some way to ensure proxy buffer doesn't take up some amount of space based on historical
// buffer allocation of the program
GPUBufferHandle MetalExecutor::access_proxy(size_t data_size) {
    NSUInteger buffer_size = [p_metal_impl->buffer_map_.at(proxy_buffer_) length];

    if (data_size > buffer_size) {
        buffer_size *= 2;
        proxy_buffer_ = allocate_buffer(buffer_size, proxy_buffer_.mem_hint);
    }

    return proxy_buffer_;
}

// TODO: How can we make command buffers more optimized for batch usages? Creating + committing each time -> perf
// overhead

// For private resources: Create a proxy buffer that can transfer to private
GPUState MetalExecutor::blit_to_private(std::span<const std::byte> data_mem, const GPUBufferHandle &buffer_handle,
                                        std::size_t data_size, bool sync) {
    id<MTLBuffer> buffer = p_metal_impl->buffer_map_.find(buffer_handle)->second;
    GPUBufferHandle proxy_buffer_handle = access_proxy(data_size);
    copy_to_device(data_mem, proxy_buffer_, data_size,
                   sync); // Should be safe since proxy will *never* be in private memory
    id<MTLBuffer> proxy_buffer = p_metal_impl->buffer_map_.at(proxy_buffer_handle);

    id<MTLCommandBuffer> transfer_cmd_buffer = [p_metal_impl->command_queue_ commandBuffer];
    id<MTLBlitCommandEncoder> blit_encoder = [transfer_cmd_buffer blitCommandEncoder];

    [blit_encoder copyFromBuffer:proxy_buffer sourceOffset:0 toBuffer:buffer destinationOffset:0 size:data_size];
    [blit_encoder endEncoding];
    [transfer_cmd_buffer commit];

    if (sync) {
        [transfer_cmd_buffer waitUntilCompleted];
    }

    return GPUState::GPUSuccess;
}

// For managed resources: Manually synchronized, CPU/GPU have unique copies
GPUState MetalExecutor::memcpy_to_managed(std::span<const std::byte> data_mem, const GPUBufferHandle &buffer_handle,
                                          std::size_t data_size) {
    id<MTLBuffer> buffer = p_metal_impl->buffer_map_.find(buffer_handle)->second;
    void *buffer_mem = [buffer contents];
    memcpy(buffer_mem, data_mem.data(), data_size);

    // Notify the GPU new memory has arrived
    [buffer didModifyRange:NSMakeRange(0, data_size)];

    return GPUState::GPUSuccess;
}

// For private resources: Create a proxy buffer that can contain private data and transfer to CPU
GPUState MetalExecutor::private_to_cpu(std::span<std::byte> data_mem, const GPUBufferHandle &buffer_handle,
                                       std::size_t data_size, bool sync) {
    id<MTLBuffer> buffer = p_metal_impl->buffer_map_.find(buffer_handle)->second;
    // Use a proxy buffer to transfer data
    GPUBufferHandle proxy_buffer_handle = access_proxy(data_size);
    copy_to_device(data_mem, proxy_buffer_, data_size, sync);
    id<MTLBuffer> proxy_buffer = p_metal_impl->buffer_map_.at(proxy_buffer_handle);

    id<MTLCommandBuffer> transfer_cmd_buffer = [p_metal_impl->command_queue_ commandBuffer];
    id<MTLBlitCommandEncoder> blit_encoder = [transfer_cmd_buffer blitCommandEncoder];

    [blit_encoder copyFromBuffer:buffer sourceOffset:0 toBuffer:proxy_buffer destinationOffset:0 size:data_size];
    [blit_encoder endEncoding];
    [transfer_cmd_buffer commit];

    [transfer_cmd_buffer addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull) {
      void *proxy_mem = [proxy_buffer contents];
      memcpy(data_mem.data(), proxy_mem, data_size);
    }];

    if (sync) {
        [transfer_cmd_buffer waitUntilCompleted];
    }

    return GPUState::GPUSuccess;
}

// For managed resources: CPU/GPU have unique copies - block to ensure data transfer is complete
GPUState MetalExecutor::managed_to_cpu(std::span<std::byte> data_mem, const GPUBufferHandle &buffer_handle,
                                       std::size_t data_size, bool sync) {
    id<MTLBuffer> buffer = p_metal_impl->buffer_map_.find(buffer_handle)->second;
    id<MTLCommandBuffer> synchronize_cmd_buffer = [p_metal_impl->command_queue_ commandBuffer];
    id<MTLBlitCommandEncoder> blit_encoder = [synchronize_cmd_buffer blitCommandEncoder];

    [blit_encoder synchronizeResource:buffer];
    [blit_encoder endEncoding];
    [synchronize_cmd_buffer commit];

    // Allows for memory copying after gpu is finished - doesn't block CPU unless asked for
    [synchronize_cmd_buffer addCompletedHandler:^(id<MTLCommandBuffer> synchronize_buffer) {
      void *buffer_mem = [buffer contents];
      memcpy(data_mem.data(), buffer_mem, data_size);
    }];

    if (sync) {
        [synchronize_cmd_buffer waitUntilCompleted];
    }

    return GPUState::GPUSuccess;
}

// TODO: Perhaps we want to have some sort of MemoryAllocator that can determine when it's best to use multiple buffers
// vs. one large one?
GPUState MetalExecutor::copy_to_device(std::span<const std::byte> data_mem, const GPUBufferHandle &buffer_handle,
                                       std::size_t data_size, bool sync) {
    auto buffer_it = p_metal_impl->buffer_map_.find(buffer_handle);
    if (buffer_it == p_metal_impl->buffer_map_.end()) {
        return GPUState::GhostBuffer;
    }

    id<MTLBuffer> buffer = buffer_it->second;

    // Ensure requested buffer has enough space
    if ([buffer length] < data_size) {
        return GPUState::GPUFailure;
    }

    // TODO: Implement shared memory
    switch (buffer_handle.mem_hint) {
    case MemoryHint::DeviceLocal:
        return blit_to_private(data_mem, buffer_handle, data_size, sync);
    case MemoryHint::HostVisible:
        if (([buffer resourceOptions] & MTLResourceStorageModeManaged) == MTLResourceStorageModeManaged) {
            return memcpy_to_managed(data_mem, buffer_handle, data_size);
        } else if (([buffer resourceOptions] & MTLResourceStorageModeShared) == MTLResourceStorageModeShared) {
            // Implement shared memory access
        }
    }

    // Invalid buffer storage type - shouldn't trigger
    return GPUState::GPUFailure;
}

GPUState MetalExecutor::copy_from_device(std::span<std::byte> data_mem, const GPUBufferHandle &buffer_handle,
                                         std::size_t data_size, bool sync) {
    auto buffer_it = p_metal_impl->buffer_map_.find(buffer_handle);
    if (buffer_it == p_metal_impl->buffer_map_.end()) {
        return GPUState::GhostBuffer;
    }

    id<MTLBuffer> buffer = buffer_it->second;

    // TODO: Implement shared memory
    switch (buffer_handle.mem_hint) {
    case MemoryHint::DeviceLocal:
        return private_to_cpu(data_mem, buffer_handle, data_size, sync);
    case MemoryHint::HostVisible:
        if (([buffer resourceOptions] & MTLResourceStorageModeManaged) == MTLResourceStorageModeManaged) {
            return managed_to_cpu(data_mem, buffer_handle, data_size, sync);
        } else if (([buffer resourceOptions] & MTLResourceStorageModeShared) == MTLResourceStorageModeShared) {
            // Implement shared memory access
        }
    }

    // Invalid buffer storage type - shouldn't trigger
    return GPUState::GPUFailure;
}

GPUState MetalExecutor::execute_batch(const std::vector<KernelDispatch> &kernels, const DispatchType &dispatch_type,
                                      std::function<void()> &cpu_callback) {
    for (const KernelDispatch &kernel : kernels) {
        // Construct a buffer for each kernel, needed to allow for end commands for individual kernel completion
        id<MTLCommandBuffer> compute_buffer = [p_metal_impl->command_queue_ commandBuffer];

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
        id<MTLComputePipelineState> compute_pipeline = p_metal_impl->find_cache_pipeline(kernel.kernel_name);

        [compute_encoder setComputePipelineState:compute_pipeline];

        for (int j = 0; j < kernel.buffer_handles.size(); ++j) {
            // TODO: Handle if buffer is not in the map (invalid buffer, NEVER ALLOCATED)
            id<MTLBuffer> bind_buffer = p_metal_impl->buffer_map_[kernel.buffer_handles[j]];

            // Each buffer must be bound at a unique index
            [compute_encoder setBuffer:bind_buffer offset:0 atIndex:j];
        }

        MTLSize groups_per_grid = MTLSizeMake(kernel.kernel_size[0], kernel.kernel_size[1], kernel.kernel_size[2]);
        MTLSize threads_per_group =
            MTLSizeMake(kernel.threads_per_group[0], kernel.threads_per_group[1], kernel.threads_per_group[2]);
        [compute_encoder dispatchThreadgroups:groups_per_grid threadsPerThreadgroup:threads_per_group];

        // Update when each kernel ends individually, rather than when the entire batch ends
        [compute_buffer addCompletedHandler:^(id<MTLCommandBuffer> compute_buffer) {
          cpu_callback();
        }];

        [compute_encoder endEncoding];
        [compute_buffer commit];
    }

    return GPUState::GPUSuccess;
}

GPUState MetalExecutor::execute_kernel(const KernelDispatch &kernel, std::function<void()> &cpu_callback) {
    return execute_batch({kernel}, DispatchType::Serial, cpu_callback);
}

// Add a final empty command buffer and block until it finishes
GPUState MetalExecutor::synchronize() {
    id<MTLCommandBuffer> synchronize_buffer = [p_metal_impl->command_queue_ commandBuffer];

    [synchronize_buffer commit];
    [synchronize_buffer waitUntilCompleted];

    return GPUState::GPUSuccess;
}

int MetalExecutor::get_buffer_length(const GPUBufferHandle &buffer_handle) {
    return [p_metal_impl->buffer_map_[buffer_handle] length];
}
