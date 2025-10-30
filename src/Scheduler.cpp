#include "Scheduler.h"
#include "DataManager.h"
#include "IGPUExecutor.h"
#include "Runtime.h"
#include "Tasks.h"
#include <algorithm>
#include <map>
#include <memory>
#include <queue>
#include <unordered_map>
#include <vector>

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
    for (int i = 0; i < gpu_task.input_ids.size(); ++i) {
        size_t input_data_size = data_manager.get_data_length(gpu_task.input_ids[i]);
        MemoryHint data_mem_hint = data_manager.get_mem_hint(gpu_task.input_ids[i]);
        auto input_data = data_manager.get_data_span(gpu_task.input_ids[i]);

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
 *
 *  Moving to implement an event-driven reaction system to avoid inefficient polling
 *  TODO: What if we instead make the Scheduler purely event driven, removing the need to wait on futures?
    //  - i.e. what if the CPU triggers a condition variable when the the task ends just like how Metal allows for
    //  completion handlers
    //      - Employ a thread safe queue that has a condition variable such that when something is added to a completion
    //      queue it wakes up and updates everything
 */
template <typename F, class... Types> void Scheduler::execute_graph(const TaskGraph &task_graph) {
    // Map that indicates each tasks current state, and ensure the root tasks (no dependencies) are ready for exec
    std::unordered_map<int, TaskRuntimeState> graph_tasks;
    int num_complete;

    std::queue<int> ready_queue;
    std::vector<int> running_tasks;

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

    while (num_complete < graph_tasks.size()) {
        // Dispatch loop - handle ready tasks
        // Shouldn't be while (!ready_queue.empty()) for a regular queue since may need to wait for CPU/GPU load to
        // decrease and don't want scheduler to hang waiting
        // Instead employ a *priority queue* implementation (i.e. a heap structure)
        // TODO: How should we order this priority queue?
        while (!ready_queue.empty()) {
            // TODO: 2. How can we effectively check for resources on CPU/GPU for scheduling?
            // Currently naive immediate scheduling approach
            int ready_task_id = ready_queue.front();
            ready_queue.pop();

            task_graph.get_task(ready_task_id)->accept(*this);
            running_tasks.push_back(ready_task_id);
        }

        // Event loop - handle efficiently waiting on running tasks
        // TODO: Refactor IGPUExector to return a future for each kernel
        if (!running_tasks.empty()) {
            // TODO: How can we efficently wait for *any* task future to be finished?
            // Currently just selects the first running task
            auto running_task_id = running_tasks.front();
            running_tasks.erase(running_tasks.begin());

            /*
             * TODO: Modify visit methods to return futures so this structure can work
             *
             * TODO: Future checking
             * Should probably implement this using condition variables instead, so this whole section will get
             * refactored
             * General idea: Maintain a list of completed task ids and switch the running tasks vector to contain
             * futures
             *  - Maintain a worker thread that checks the running tasks for future completions and wakes up the main
             * thread via a cond. var, pushing the completed task to the complete queue
             *  - Core loop then runs dependency removal
             */

            num_complete++;
            graph_tasks[running_task_id].state = TaskState::Complete;
            for (int dependent_id : task_graph.get_dependents(running_task_id)) {
                graph_tasks[dependent_id].num_dependencies--;

                if (graph_tasks[dependent_id].num_dependencies == 0) {
                    ready_queue.push(dependent_id);
                }
            }
        }
    }

    // Inefficient polling solution
    /*
        while (num_complete < graph_tasks.size()) {
            for (auto task_iter = graph_tasks.begin(); task_iter != graph_tasks.end(); ++task_iter) {

                if (task_iter->second.state == TaskState::Ready) {
                    task_graph.get_task(task_iter->first)->accept(*this);
                }

                if (task_iter->second.state == TaskState::Running) {
                    std::shared_ptr<ITask> running_task = task_graph.get_task(task_iter->first);

                    if (auto gpu_task = std::dynamic_pointer_cast<GPUTask>(running_task)) {
                        if (gpu_executor->get_kernel_status(gpu_task->task_name)) {
                            task_iter->second.state = TaskState::Complete;
                            num_complete++;

                            for (int dependent_id : task_graph.get_dependents(task_iter->first)) {
                                graph_tasks[dependent_id].num_dependencies -= 1;
                            }
                        }
                    } else if (auto cpu_task = std::dynamic_pointer_cast<CPUTask<F, Types...>>(running_task)) {
                        if (cpu_task->task_complete()) {
                            task_iter->second.state = TaskState::Complete;
                            num_complete++;

                            for (int dependent_id : task_graph.get_dependents(task_iter->first)) {
                                graph_tasks[dependent_id].num_dependencies -= 1;
                            }
                        }
                    }
                }
            }
        }
    */
}
