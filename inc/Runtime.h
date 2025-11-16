#ifndef RUNTIME_H
#define RUNTIME_H

#include "DataManager.h"
#include "IGPUExecutor.h"
#include "Tasks.h"
#include "ThreadPool.h"
#include <memory>
#include <variant>

enum class TaskState { Pending, Ready, Running, Complete };

struct TaskRuntimeState {
    TaskState state;
    int num_dependencies;
};

enum class GPUBackend { Metal, Cuda };

struct GPUDevice {
    GPUBackend backend;
    // implicit GPU to use (e.g. Metal), otherwise an ID should be provided (e.g. CUDA)
    int device_id;

    GPUDevice(GPUBackend backend, int device_id = -1) : backend(backend), device_id(device_id) {};
};

/*
 * The Runtime is the owner of all the system resources
 * It coordinates data handling and creating schedulers
 */
class Runtime {
  public:
    // The GPU executor and Thread Pool are created at initialization to accurately reflect system state
    Runtime(DataManager &data_manager, size_t num_threads) : data_manager_(data_manager), num_threads(num_threads) {};

    // Immediately communicates with the scheduler to begin executing tasks
    std::future<void> commit_graph(TaskGraph &task_graph, GPUDevice &device_info);

  private:
    DataManager &data_manager_;
    std::unique_ptr<ThreadPool> thread_pool_;
    std::unique_ptr<IGPUExecutor> gpu_exec_;
    size_t num_threads;

    void create_thread_pool_() { thread_pool_ = std::make_unique<ThreadPool>(num_threads); };
    void create_executor_(GPUDevice &device_info, const TaskGraph &task_graph);
};

#endif
