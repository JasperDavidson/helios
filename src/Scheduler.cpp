#include "Scheduler.h"
#include "DataManager.h"
#include "IGPUExecutor.h"
#include "Runtime.h"
#include "Tasks.h"
#include <algorithm>
#include <map>
#include <memory>
#include <unordered_map>

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
template <typename F, class... Types> void Scheduler::execute_graph(const TaskGraph &task_graph) {
    // Map that indicates each tasks current state, and ensure the root tasks (no dependencies) are ready for exec
    std::unordered_map<int, TaskRuntimeState> graph_tasks;
    for (int task : task_graph.get_task_ids()) {
        TaskState task_state;
        int num_dependencies = task_graph.get_dependencies(task).size();
        if (num_dependencies == 0) {
            task_state = TaskState::Ready;
        } else {
            task_state = TaskState::Pending;
        }

        graph_tasks[task] = TaskRuntimeState(task_state, num_dependencies);
    }

    while (/* how to efficiently find if not all tasks are complete? */ true) {
        // 1.

        // 2.

        // 3.
        for (auto task_iter = graph_tasks.begin(); task_iter != graph_tasks.end(); ++task_iter) {
            if (task_iter->second.state == TaskState::Running) {
                std::shared_ptr<ITask> running_task = task_graph.get_task(task_iter->first);

                if (auto gpu_task = std::dynamic_pointer_cast<GPUTask>(running_task)) {
                } else if (auto cpu_task = std::dynamic_pointer_cast<CPUTask<F, Types...>>(running_task)) {
                    if (cpu_task->task_complete()) {
                        task_iter->second.state = TaskState::Complete;

                        for (int dependent_id : task_graph.get_dependents(task_iter->first)) {
                            graph_tasks[dependent_id].num_dependencies -= 1;
                        }
                    }
                }
            }
        }
    }
}
