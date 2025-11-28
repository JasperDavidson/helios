#include "MetalExecutor.h"
#include "DataManager.h"
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
#include <memory>
#include <objc/NSObjCRuntime.h>
#include <stdexcept>
#include <unordered_map>

static constexpr NSString *const LIBRARY_NAME = @"kernels";

struct MetalExecutor::MetalExecutorImpl {
    // Metal specific variables
    id<MTLDevice> mtl_device_;
    id<MTLCommandQueue> command_queue_;
    id<MTLLibrary> mtl_library_;

    // Slab buffers for buddy memory allocation
    id<MTLBuffer> devloc_slab_buffer_;
    id<MTLBuffer> hostvis_slab_buffer_;
    id<MTLBuffer> unified_slab_buffer_;

    // map for storing already prepared compute pipelines
    std::unordered_map<std::string, id<MTLComputePipelineState>> pipeline_map_;

    // map for storing relating synchronization events to their (shared memory) buffers
    std::unordered_map<GPUBufferHandle, id<MTLSharedEvent>> shared_event_map_;

    // map for storing GPU buffer handles
    std::unordered_map<GPUBufferHandle, id<MTLBuffer>> buffer_map_;

    id<MTLBuffer> select_buffer(MemoryHint mem_hint) {
        switch (mem_hint) {
        case MemoryHint::DeviceLocal:
            return devloc_slab_buffer_;
            break;
        case MemoryHint::Unified:
            return unified_slab_buffer_;
            break;
        case MemoryHint::HostVisible:
            return hostvis_slab_buffer_;
            break;
        default:
            throw std::runtime_error("Tried to fetch an invalid buffer!");
        }
    }

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

MetalExecutor::MetalExecutor(std::pair<int, int> devloc_bounds, std::pair<int, int> hostvis_bounds,
                             std::pair<int, int> unified_bounds, size_t proxy_size)
    : p_metal_impl(std::make_unique<MetalExecutorImpl>()) {
    // Construct the GPUMemoryAllocator
    mem_allocator = GPUMemoryAllocator(devloc_bounds.first, devloc_bounds.second, unified_bounds.first,
                                       unified_bounds.second, hostvis_bounds.first, hostvis_bounds.second);

    if (devloc_bounds != std::pair(0, 0)) {
        size_t devloc_size = devloc_bounds.second - devloc_bounds.first;
        p_metal_impl->devloc_slab_buffer_ =
            [p_metal_impl->mtl_device_ newBufferWithLength:devloc_size options:MTLResourceStorageModePrivate];
    }

    if (hostvis_bounds != std::pair(0, 0)) {
        size_t hostvis_size = hostvis_bounds.second - hostvis_bounds.first;
        p_metal_impl->hostvis_slab_buffer_ =
            [p_metal_impl->mtl_device_ newBufferWithLength:hostvis_size options:MTLResourceStorageModeManaged];
    }

    if (unified_bounds != std::pair(0, 0)) {
        size_t unified_size = unified_bounds.second - unified_bounds.first;
        p_metal_impl->unified_slab_buffer_ =
            [p_metal_impl->mtl_device_ newBufferWithLength:unified_size options:MTLResourceStorageModeShared];
    }

    p_metal_impl->mtl_device_ = MTLCreateSystemDefaultDevice();
    if (!p_metal_impl->mtl_device_) {
        throw std::runtime_error("Failed to get Metal device.");
    }
    p_metal_impl->command_queue_ = [p_metal_impl->mtl_device_ newCommandQueue];
    load_default_library();

    p_metal_impl->pipeline_map_ = std::unordered_map<std::string, id<MTLComputePipelineState>>();

    if (proxy_size > 0) {
        // TODO: Make the free map hierarchical
        // Currently the free map will find addresses outside of the order that was expected
        proxy_handle_ = allocate_buffer(proxy_size, MemoryHint::Unified);
        GPUBufferHandle buffer_test = allocate_buffer(32, MemoryHint::Unified);
        GPUBufferHandle buffer_test2 = allocate_buffer(32, MemoryHint::Unified);
        GPUBufferHandle buffer_test3 = allocate_buffer(32, MemoryHint::Unified);
        GPUBufferHandle buffer_test4 = allocate_buffer(64, MemoryHint::Unified);
        mem_allocator.check_free_mem(proxy_handle_.size, proxy_handle_.mem_offset, proxy_handle_.mem_hint);
        mem_allocator.check_free_mem(buffer_test2.size, buffer_test2.mem_offset, buffer_test2.mem_hint);
        mem_allocator.check_free_mem(buffer_test.size, buffer_test.mem_offset, buffer_test.mem_hint);
        mem_allocator.check_free_mem(buffer_test3.size, buffer_test3.mem_offset, buffer_test3.mem_hint);
        mem_allocator.check_free_mem(buffer_test4.size, buffer_test4.mem_offset, buffer_test4.mem_hint);
        proxy_handle_ = allocate_buffer(proxy_size, MemoryHint::Unified);
    }
}

// NOTE: This destructor may need to be filled in (e.g. deallocating slab buffers)
MetalExecutor::~MetalExecutor() = default;

GPUBufferHandle MetalExecutor::allocate_buffer(std::size_t buffer_size, const MemoryHint mem_hint) {
    // Create the buffer handle object
    std::cout << "Allocating memory..." << std::endl;
    size_t free_offset = mem_allocator.allocate_memory(buffer_size, mem_hint);
    std::cout << "Allocated memory!" << " buffer offset: " << free_offset << std::endl;
    GPUBufferHandle buffer_handle(buffer_counter, mem_hint, free_offset, buffer_size);
    buffer_counter++;

    return buffer_handle;
}

GPUState MetalExecutor::deallocate_buffer(const GPUBufferHandle &buffer_handle) {
    mem_allocator.check_free_mem(buffer_handle.mem_offset, buffer_handle.size, buffer_handle.mem_hint);

    // Remove the reference to the command buffer - Metal will automatically deallocate resources
    //    if (p_metal_impl->buffer_map_.contains(buffer_handle)) {
    //        p_metal_impl->buffer_map_.erase(buffer_handle);
    //
    //        return GPUState::GPUSuccess;
    //    }

    return GPUState::GPUSuccess;
}

void MetalExecutor::access_proxy(size_t data_size) {
    if (data_size > proxy_handle_.size) {
        mem_allocator.check_free_mem(proxy_handle_.size, 0, MemoryHint::Unified);
        proxy_handle_.size *= 2;
        mem_allocator.allocate_memory(proxy_handle_.size, MemoryHint::Unified);
    }
}

// TODO: How can we make command buffers more optimized for batch usages? Creating + committing each time -> perf
// overhead

// For private resources: Create a proxy buffer that can transfer to private
GPUState MetalExecutor::blit_to_private(std::span<const std::byte> data_mem, const GPUBufferHandle &buffer_handle) {
    access_proxy(buffer_handle.size);
    copy_to_device(data_mem, proxy_handle_); // Should be safe since proxy will *never* be in private memory

    id<MTLCommandBuffer> transfer_cmd_buffer = [p_metal_impl->command_queue_ commandBuffer];
    id<MTLBlitCommandEncoder> blit_encoder = [transfer_cmd_buffer blitCommandEncoder];

    [blit_encoder copyFromBuffer:p_metal_impl->unified_slab_buffer_
                    sourceOffset:proxy_handle_.mem_offset
                        toBuffer:p_metal_impl->devloc_slab_buffer_
               destinationOffset:buffer_handle.mem_offset
                            size:buffer_handle.size];
    [blit_encoder endEncoding];

    [transfer_cmd_buffer waitUntilCompleted];
    [transfer_cmd_buffer commit];

    return GPUState::GPUSuccess;
}

// For managed resources: Manually synchronized, CPU/GPU have unique copies
GPUState MetalExecutor::copy_to_managed(std::span<const std::byte> data_mem, const GPUBufferHandle &buffer_handle) {
    //    id<MTLBuffer> buffer = p_metal_impl->buffer_map_.find(buffer_handle)->second;
    //    void *buffer_mem = [buffer contents];

    void *slab_mem = [p_metal_impl->devloc_slab_buffer_ contents];
    void *buffer_mem = (char *)slab_mem + buffer_handle.mem_offset;
    memcpy(buffer_mem, data_mem.data(), buffer_handle.size);

    // Notify the GPU new memory has arrived
    [p_metal_impl->devloc_slab_buffer_
        didModifyRange:NSMakeRange(*(char *)buffer_mem, *(char *)buffer_mem + buffer_handle.size)];

    return GPUState::GPUSuccess;
}

GPUState MetalExecutor::copy_to_shared(std::span<const std::byte> data_mem, const GPUBufferHandle &buffer_handle) {
    //    id<MTLBuffer> buffer = p_metal_impl->buffer_map_.find(buffer_handle)->second;
    //    void *buffer_mem = [buffer contents];

    void *slab_mem = [p_metal_impl->unified_slab_buffer_ contents];
    void *buffer_mem = (char *)slab_mem + buffer_handle.mem_offset;
    memcpy(buffer_mem, data_mem.data(), buffer_handle.size);

    return GPUState::GPUSuccess;
}

// For private resources: Create a proxy buffer that can contain private data and transfer to CPU
GPUState MetalExecutor::private_to_cpu(std::span<std::byte> data_mem, const GPUBufferHandle &buffer_handle) {
    access_proxy(buffer_handle.size);

    id<MTLCommandBuffer> transfer_cmd_buffer = [p_metal_impl->command_queue_ commandBuffer];
    id<MTLBlitCommandEncoder> blit_encoder = [transfer_cmd_buffer blitCommandEncoder];

    [blit_encoder copyFromBuffer:p_metal_impl->devloc_slab_buffer_
                    sourceOffset:buffer_handle.mem_offset
                        toBuffer:p_metal_impl->unified_slab_buffer_
               destinationOffset:proxy_handle_.mem_offset
                            size:buffer_handle.size];
    [blit_encoder endEncoding];

    [transfer_cmd_buffer addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull) {
      void *slab_mem = [p_metal_impl->devloc_slab_buffer_ contents];
      void *proxy_mem = (char *)slab_mem + proxy_handle_.mem_offset;

      memcpy(data_mem.data(), proxy_mem, buffer_handle.size);
    }];
    [transfer_cmd_buffer waitUntilCompleted];
    [transfer_cmd_buffer commit];

    return GPUState::GPUSuccess;
}

// For managed resources: CPU/GPU have unique copies - block to ensure data transfer is complete
GPUState MetalExecutor::managed_to_cpu(std::span<std::byte> data_mem, const GPUBufferHandle &buffer_handle) {
    id<MTLBuffer> buffer = p_metal_impl->hostvis_slab_buffer_;

    id<MTLCommandBuffer> synchronize_cmd_buffer = [p_metal_impl->command_queue_ commandBuffer];
    id<MTLBlitCommandEncoder> blit_encoder = [synchronize_cmd_buffer blitCommandEncoder];

    [blit_encoder synchronizeResource:buffer];
    [blit_encoder endEncoding];
    [synchronize_cmd_buffer addCompletedHandler:^(id<MTLCommandBuffer> synchronize_buffer) {
      void *slab_mem = [buffer contents];
      void *buffer_mem = (char *)slab_mem + buffer_handle.mem_offset;
      memcpy(data_mem.data(), buffer_mem, buffer_handle.size);
    }];

    [synchronize_cmd_buffer commit];

    return GPUState::GPUSuccess;
}

GPUState MetalExecutor::shared_to_cpu(std::span<std::byte> data_mem, const GPUBufferHandle &buffer_handle) {
    //    id<MTLBuffer> buffer = p_metal_impl->buffer_map_.find(buffer_handle)->second;
    //    void *buffer_mem = [buffer contents];

    void *slab_mem = [p_metal_impl->unified_slab_buffer_ contents];
    void *buffer_mem = (char *)slab_mem + buffer_handle.mem_offset;
    memcpy(data_mem.data(), buffer_mem, buffer_handle.size);

    return GPUState::GPUSuccess;
}

// TODO: Perhaps we want to have some sort of MemoryAllocator that can determine when it's best to use multiple buffers
// vs. one large one?
GPUState MetalExecutor::copy_to_device(std::span<const std::byte> data_mem, const GPUBufferHandle &buffer_handle) {
    //    auto buffer_it = p_metal_impl->buffer_map_.find(buffer_handle);
    //    if (buffer_it == p_metal_impl->buffer_map_.end()) {
    //        return GPUState::GhostBuffer;
    //    }
    //
    //    id<MTLBuffer> buffer = buffer_it->second;
    //
    //    // Ensure requested buffer has enough space
    //    if (buffer_handle.size < data_size) {
    //        return GPUState::GPUFailure;
    //    }

    switch (buffer_handle.mem_hint) {
    case MemoryHint::DeviceLocal:
        return blit_to_private(data_mem, buffer_handle);
    case MemoryHint::HostVisible:
        return copy_to_managed(data_mem, buffer_handle);
    case MemoryHint::Unified:
        return copy_to_shared(data_mem, buffer_handle);
    }

    // Invalid buffer storage type - shouldn't trigger
    return GPUState::GPUFailure;
}

GPUState MetalExecutor::copy_from_device(std::span<std::byte> data_mem, const GPUBufferHandle &buffer_handle) {
    //    auto buffer_it = p_metal_impl->buffer_map_.find(buffer_handle);
    //    if (buffer_it == p_metal_impl->buffer_map_.end()) {
    //        return GPUState::GhostBuffer;
    //    }
    //
    //    id<MTLBuffer> buffer = buffer_it->second;

    switch (buffer_handle.mem_hint) {
    case MemoryHint::DeviceLocal:
        return private_to_cpu(data_mem, buffer_handle);
    case MemoryHint::HostVisible:
        return managed_to_cpu(data_mem, buffer_handle);
    case MemoryHint::Unified:
        return shared_to_cpu(data_mem, buffer_handle);
    }

    // Invalid buffer storage type - shouldn't trigger
    return GPUState::GPUFailure;
}

// See if we can optimize this further -> creating a new compute buffer each time is bad perf but also need cpu
// callbacks
GPUState MetalExecutor::execute_batch(const std::vector<KernelDispatch> &kernels, const DispatchType &dispatch_type,
                                      std::function<void()> &cpu_callback) {
    for (const KernelDispatch &kernel : kernels) {
        // Construct a buffer for each kernel, needed to allow for end commands for individual kernel completion
        id<MTLCommandBuffer> compute_buffer = [p_metal_impl->command_queue_ commandBuffer];

        // Prepare the compute encoder (binding GPU resources, providing compute pipeline, etc.)
        id<MTLComputeCommandEncoder> compute_encoder;

        switch (dispatch_type) {
        case DispatchType::Serial:
            compute_encoder = [compute_buffer computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
            break;
        case DispatchType::Concurrent:
            compute_encoder = [compute_buffer computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
            break;
        default:
            return GPUState::InvalidDispatchType;
        }

        // Fetch the compute pipeline or create it if needed
        id<MTLComputePipelineState> compute_pipeline = p_metal_impl->find_cache_pipeline(kernel.kernel_name);

        [compute_encoder setComputePipelineState:compute_pipeline];

        for (int j = 0; j < kernel.buffer_handles.size(); ++j) {
            id<MTLBuffer> slab_bind_buffer = p_metal_impl->select_buffer(kernel.buffer_handles[j].mem_hint);

            // Each buffer must be bound at a unique index
            [compute_encoder setBuffer:slab_bind_buffer offset:kernel.buffer_handles[j].mem_offset atIndex:j];
        }

        MTLSize groups_per_grid = MTLSizeMake(kernel.grid_dim[0], kernel.grid_dim[1], kernel.grid_dim[2]);
        MTLSize threads_per_group = MTLSizeMake(kernel.block_dim[0], kernel.block_dim[1], kernel.block_dim[2]);
        [compute_encoder dispatchThreadgroups:groups_per_grid threadsPerThreadgroup:threads_per_group];

        // Update when each kernel ends individually, rather than when the entire batch ends
        // Might cause perf issues - how could we improve this?
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
