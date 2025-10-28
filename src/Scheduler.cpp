#include "Scheduler.h"
#include "DataManager.h"
#include "IGPUExecutor.h"
#include "Tasks.h"
#include <algorithm>
#include <future>
#include <map>

void Scheduler::visit(const GPUTask &gpu_task) const {
    // Figure out if appropriate buffer space already exists on the GPU; if not create it
    // - This allows for reusing buffers from previous kernels, saving resources
    std::vector<GPUBufferHandle> buffers_not_in_use; //  = someAlgorithmToDetermineBuffersNotInUse
    std::multimap<int, GPUBufferHandle> size_to_buffer;

    // Use multimap for better efficiency when searching for unused buffers
    std::ranges::for_each(buffers_not_in_use, [&](const GPUBufferHandle &buffer_handle) {
        size_to_buffer.emplace(data_manager.get_data_length(buffer_handle.ID), buffer_handle);
    });

    std::vector<BufferBinding> buffer_bindings;
    for (int i = 0; i < gpu_task.inputs.size(); ++i) {
        size_t input_data_size = data_manager.get_data_length(gpu_task.inputs[i]->ID);
        MemoryHint data_mem_hint = data_manager.get_mem_hint(gpu_task.inputs[i]->ID);
        auto input_data = data_manager.get_data_span(gpu_task.inputs[i]->ID);

        auto potential_buffer = size_to_buffer.lower_bound(input_data_size);

        if (potential_buffer != size_to_buffer.end()) {
            potential_buffer->second.mem_hint = data_mem_hint;
            buffer_bindings.push_back(BufferBinding(potential_buffer->second, gpu_task.buffer_usages[i]));
            gpu_executor->copy_to_device(input_data, potential_buffer->second, input_data_size);
            size_to_buffer.erase(potential_buffer);
        } else {
            GPUBufferHandle new_buffer = gpu_executor->allocate_buffer(input_data_size, data_mem_hint);
            gpu_executor->copy_to_device(input_data, new_buffer, input_data_size);
        }
    }

    // Assemble the kernel dispatch and assign it to the GPU
    KernelDispatch kernel(gpu_task.task_name, buffer_bindings, gpu_task.kernel_size, gpu_task.threads_per_group);
    gpu_executor->execute_kernel(kernel);
}

/*
 * Maintains state of currently running tasks, tasks that are ready to be ran, and tasks that need dependencies met
 * Cycles through the following until all tasks are completed:
 *  1. Check all inert tasks and see if their dependencies have been met (if so add to ready_tasks)
 *  2. Check all ready tasks and attempt to schedule on CPU/GPU if appropriate resources
 *  3. Check all running tasks and see if they have completed (if so remove them as dependencies)
 */
void Scheduler::execute_graph(const TaskGraph &task_graph) {
    // Tasks that have their dependencies met but aren't currently running
    std::vector<int> ready_tasks = task_graph.find_ready();
    // Tasks that are currently running and need to be checked for completion
    std::future<void> running_tasks;
}
