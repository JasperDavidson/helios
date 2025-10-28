#ifndef RUNTIME_H
#define RUNTIME_H

#include "DataManager.h"
#include "IGPUExecutor.h"
#include "Tasks.h"
#include "ThreadPool.h"

enum class TaskState { Pending, Ready, Running, Complete };

struct TaskRuntimeState {
    TaskState state;
    int num_dependencies;
};

/*
 * The Runtime is the owners of all the system resources
 * It coordinates data handling and creating schedulers
 */
class Runtime {
  public:
    // Returns a future so that user can wait for the graph to complete when needed
    // Immediately communicatoes with the scheduler to begin executing tasks
    std::future<void> commit_graph(TaskGraph &task_graph);

  private:
    DataManager data_manager;
    std::shared_ptr<ThreadPool> thread_pool;
    std::shared_ptr<IGPUExecutor> gpu_exec;
};

#endif
