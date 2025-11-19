#include "Scheduler.h"
#include "DataManager.h"
#include "IGPUExecutor.h"
#include "Runtime.h"
#include "Tasks.h"
#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

const size_t COUNTER_BUFFER_SIZE = 8;

size_t count_bytes_to_size(const std::span<std::byte> bytes) {
    size_t val = 0;

    for (int i = 0; i < bytes.size(); ++i) {
        val |= (static_cast<size_t>(bytes[i]) << (i * 8));
    }

    return val;
}

// URGENT: Look into actually tracking CPU return future
// Just ake sure "lambda with completion" idea actually makes sense
void Scheduler::visit(const BaseCPUTask &cpu_task) {
    auto lambda_with_completion = [this, &cpu_task] {
        cpu_task.task_lambda();
        completed_queue.push_task(cpu_task.id);
    };

    thread_pool->add_task(lambda_with_completion);
};

void Scheduler::visit(const GPUTask &gpu_task) {
    // Figure out if appropriate buffer space already exists on the GPU; if not create it
    // - This allows for reusing buffers from previous kernels, saving resources
    std::vector<GPUBufferHandle> buffers_not_in_use; // TODO:  = someAlgorithmToDetermineBuffersNotInUse
    std::multimap<int, GPUBufferHandle> size_to_buffer;

    // Use multimap for better efficiency when searching for unused buffers
    for (const GPUBufferHandle &buffer_handle : buffers_not_in_use) {
        size_to_buffer.emplace(data_manager.get_data_length(buffer_handle.id), buffer_handle);
    }

    size_t max_input_size = 0;
    std::vector<GPUBufferHandle> buffer_handles;
    for (int i = 0; i < gpu_task.input_ids.size(); ++i) {
        int data_id = gpu_task.input_ids[i];
        if (gpu_executor->data_buffer_exists(data_id)) {
            continue;
        }

        size_t input_data_size = data_manager.get_data_length(data_id);
        MemoryHint data_mem_hint = data_manager.get_mem_hint(data_id);
        auto input_data = data_manager.get_span(data_id);

        // Maintain max input size if user doesn't specify other method for output size tracking
        max_input_size = std::max(max_input_size, input_data_size);

        auto potential_buffer = size_to_buffer.lower_bound(input_data_size);

        // TODO: Look into decoupling so the scheduler can overlay I/O and compute
        // Current serialization model is inefficient

        GPUBufferHandle buffer_in_use;
        if (potential_buffer != size_to_buffer.end()) {
            buffer_in_use = potential_buffer->second;
            buffer_in_use.mem_hint = data_mem_hint;
            size_to_buffer.erase(potential_buffer);
        } else {
            buffer_in_use = gpu_executor->allocate_buffer(input_data_size, data_mem_hint);
        }

        buffer_handles.push_back(buffer_in_use);
        gpu_executor->copy_to_device(input_data, buffer_in_use, input_data_size, true);
        gpu_executor->map_data_to_buffer(data_id, buffer_in_use);
    }

    // TODO: Should handle if output was marked as DeviceLocal and output is only intermediary for other GPU operation
    //  - Would mean data wouldn't need to copied back to CPU
    //  - Maybe could even apply an optimization to detect this?

    // Since output size is by default 0, assume user wanted max input size if it is
    size_t user_output_size = data_manager.get_data_length(gpu_task.output_id);
    size_t output_size = user_output_size == 0 ? max_input_size : user_output_size;
    MemoryHint output_mem_hint = data_manager.get_mem_hint(gpu_task.output_id);

    auto potential_output_buffer = size_to_buffer.lower_bound(output_size);
    if (potential_output_buffer != size_to_buffer.end()) {
        potential_output_buffer->second.mem_hint = output_mem_hint;
        buffer_handles.push_back(potential_output_buffer->second);
        size_to_buffer.erase(potential_output_buffer);
    } else {
        GPUBufferHandle new_buffer = gpu_executor->allocate_buffer(max_input_size, output_mem_hint);
        buffer_handles.push_back(new_buffer);
    }

    GPUBufferHandle output_buffer = buffer_handles.back();

    // Manage count buffer if requested, last buffer since it may or may not be included
    GPUBufferHandle count_buffer;
    if (gpu_task.count_buffer_active) {
        // This allows for 8 bytes of counting (64 bit size_t)
        count_buffer = gpu_executor->allocate_buffer(COUNTER_BUFFER_SIZE, MemoryHint::HostVisible);
        buffer_handles.push_back(count_buffer);
    }

    // TODO: Implement CPU callbacks for output retrieval to CPU
    // Needs to notify the event based scheduling loop once data has been fully received
    // What if the data can stay on the GPU between tasks, this would reduce overhead -> handling multiple
    // types of callbacks?

    // In general *always* returning the computed back to the CPU is ineffecient
    // Should instead return an event that signals when the computation is done and data can be fetched if desired
    std::function<void()> cpu_callback = [&]() {
        if (gpu_task.count_buffer_active) {
            // Find the number of bytes used for GPU output
            std::vector<std::byte> byte_vec(COUNTER_BUFFER_SIZE);
            std::span<std::byte> counted_span(byte_vec);
            gpu_executor->copy_from_device(counted_span, count_buffer, COUNTER_BUFFER_SIZE, true);
            size_t counted_bytes = count_bytes_to_size(counted_span);

            // Allocate memory/span on CPU for GPU output
            // This is the earliest we could possibly form the span since only now we know the size
            std::byte byte_array[counted_bytes];
            std::span<std::byte> output_span(byte_array, counted_bytes);
            this->gpu_executor->copy_from_device(output_span, count_buffer, counted_bytes, false);
        } else {
            std::span<std::byte> output_span = data_manager.get_span_mut(gpu_task.output_id);
            this->gpu_executor->copy_from_device(output_span, output_buffer, output_size, false);
        }

        completed_queue.push_task(gpu_task.id);
    };

    // Assemble the kernel dispatch and assign it to the GPU
    KernelDispatch kernel(gpu_task.task_name, buffer_handles, gpu_task.kernel_size, gpu_task.threads_per_group);
    gpu_executor->execute_kernel(kernel, cpu_callback);
};

/*
 * Maintains state of currently running tasks, tasks that are ready to be ran, and tasks that need dependencies met
 * Cycles through the following until all tasks are completed:
 *  1. Check all inert tasks and see if their dependencies have been met (if so add to ready_tasks)
 *  2. Check all ready tasks and attempt to schedule on CPU/GPU if appropriate resources
 *  3. Check all running tasks and see if they have completed (if so remove them as dependencies)
 *
 *  Moving to implement an event-driven reaction system to avoid inefficient polling
 *
 * TODO: Figure out how to incorporate Memory usages into scheduler
 *  - E.g. if memory is read/write only how can we apply optimizations?
 *
 *  TODO: How should we implement priority among tasks?
 *
 *  TODO: What if we instead make the Scheduler purely event driven, removing the need to wait on futures?
    //  - i.e. what if the CPU triggers a condition variable when the the task ends just like how Metal allows for
    //  completion handlers
    //      - Employ a thread safe queue that has a condition variable such that when something is added to a completion
    //      queue it wakes up and updates everything
 */
void Scheduler::execute_graph(const TaskGraph &task_graph) {
    // Map that indicates each tasks current state, and ensure the root tasks (no dependencies) are ready for exec
    std::unordered_map<int, TaskRuntimeState> graph_tasks;
    int num_complete = 0;

    std::queue<int> ready_queue;

    // Goal is to use this for some sort of real-time monitoring of the system
    // (Vector might not be best, but we should keep track of in-flight tasks for this)
    std::unordered_set<int> running_tasks;

    for (int task_id : task_graph.get_task_ids()) {
        TaskState task_state;
        int num_dependencies = task_graph.get_dependencies(task_id).size();
        if (num_dependencies == 0) {
            task_state = TaskState::Ready;
            ready_queue.push(task_id);
        } else {
            task_state = TaskState::Pending;
        }

        graph_tasks[task_id] = TaskRuntimeState(task_state, num_dependencies);
    }

    while (num_complete < graph_tasks.size()) {
        // Dispatch loop - handle ready tasks
        // Shouldn't be while (!ready_queue.empty()) for a regular queue since may need to wait for CPU/GPU load to
        // decrease and don't want scheduler to hang waiting
        while (!ready_queue.empty()) {
            // TODO: 2. How can we effectively check for resources on CPU/GPU for scheduling?
            // Currently naive immediate scheduling approach
            int ready_task_id = ready_queue.front();
            ready_queue.pop();

            task_graph.get_task(ready_task_id)->accept(*this);
            running_tasks.insert(ready_task_id);
        }

        // Prevents inefficient use of cycles on constant polling
        // Allows us to choose the most recent task that finished
        std::unique_lock<std::mutex> queue_lock = completed_queue.wait();
        while (!completed_queue.data_queue.empty()) {
            int completed_task = completed_queue.data_queue.front();
            completed_queue.data_queue.pop();
            num_complete++;

            graph_tasks[completed_task].state = TaskState::Complete;
            running_tasks.erase(completed_task);
            for (int dependent_id : task_graph.get_dependents(completed_task)) {
                graph_tasks[dependent_id].num_dependencies--;

                if (graph_tasks[dependent_id].num_dependencies == 0) {
                    ready_queue.push(dependent_id);
                }
            }
        }
        queue_lock.unlock();
    }
}
