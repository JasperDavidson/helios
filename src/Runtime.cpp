#include "Runtime.h"
#include "DataManager.h"
#include "MetalExecutor.h"
#include "Scheduler.h"
#include "Tasks.h"
#include <algorithm>
#include <future>
#include <memory>
#include <stdexcept>

// Allows for GPU setup before execution of tasks
void Runtime::create_executor_(GPUDevice &device_info, const TaskGraph &task_graph) {
    if (device_info.backend == GPUBackend::Metal) {
        // Finds plausible default size for proxy buffer
        std::vector<DataEntry> device_local_tasks = data_manager_.get_device_local_tasks();

        if (device_local_tasks.empty()) {
            gpu_exec_ = std::make_unique<MetalExecutor>(0);
        }

        size_t max_local_task_size = 0;
        for (DataEntry &local_task : device_local_tasks) {
            size_t local_task_size = local_task.byte_size;
            max_local_task_size = std::max(max_local_task_size, local_task_size);
        }

        gpu_exec_ = std::make_unique<MetalExecutor>(max_local_task_size);
    } else if (device_info.backend == GPUBackend::Cuda) {
        // TODO: Impl once cuda is implemented
    } else {
        std::runtime_error("Attempted to select a backend not current supported");
    }
}

// TODO: Figure out return type here - Maybe a future?
std::future<void> Runtime::commit_graph(TaskGraph &task_graph, GPUDevice &device_info) {
    task_graph.validate_graph();
    create_executor_(device_info, task_graph);
    create_thread_pool_();

    Scheduler graph_scheduler = Scheduler(data_manager_, thread_pool_, gpu_exec_);
    graph_scheduler.execute_graph(task_graph);
};
