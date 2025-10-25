#ifndef RUNTIME_H
#define RUNTIME_H

#include "DataManager.h"
#include "IGPUExecutor.h"
#include "Tasks.h"
#include "ThreadPool.h"

/*
 * The Runtime is the owners of all the system resources
 * It coordinates data handling and creating schedulers
 */
class Runtime {
  public:
    // Should return a future of the final result?
    std::shared_ptr<BaseDataHandle> commit_graph(const TaskGraph &task_graph);

  private:
    DataManager data_manager;
    std::shared_ptr<ThreadPool> thread_pool;
    std::shared_ptr<IGPUExecutor> gpu_exec;
};

#endif
