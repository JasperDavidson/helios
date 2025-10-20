#include "Scheduler.h"
#include "DataManager.h"
#include "IGPUExecutor.h"
#include "Tasks.h"

void Scheduler::visit(const GPUTask &gpu_task) {
    // 1. Figure out if appropriate buffer space already exists on the GPU; if not create it
    // - This allows for reusing buffers from previous kernels, saving resources

    std::vector<GPUBufferHandle> buffers_not_in_use; //  = someAlgorithmToDetermineBuffersNotInUse
    std::vector<BufferBinding> buffer_bindings;
    // Use ascended sort so smallest possible buffers are used first
    // std::sort(buffers_not_in_use.begin(), buffers_not_in_use.end());
    for (int i = 0; i < gpu_task.inputs.size(); ++i) {
        size_t input_data_size = data_manager.get_data_length(gpu_task.inputs[i]->ID);

        for (auto it = buffers_not_in_use.begin(); it != buffers_not_in_use.end();) {
            if (buffers_not_in_use.empty()) {
                break;
            }

            size_t unused_buffer_size = data_manager.get_data_length(it->ID);
            if (input_data_size <= unused_buffer_size) {
                // TODO: Need some way for the scheduler to include a memory hint for the buffer
                buffer_bindings.push_back(BufferBinding(*it, gpu_task.buffer_usages[i]));
                // TODO: How will we access the data mem?
                gpu_executor->copy_to_device(std::span<const std::byte> data_mem, *it, input_data_size);
                buffers_not_in_use.erase(it);
            }
        }

        // TODO: If not suitable buffer is found for reuse
    }

    // Assemble the kernel dispatch for the kernel and assign it to the GPU
}
