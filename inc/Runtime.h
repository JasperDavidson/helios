#ifndef RUNTIME_H
#define RUNTIME_H

#include "DataManager.h"
#include "IGPUExecutor.h"
#include "Tasks.h"
#include "ThreadPool.h"
#include <any>
#include <unordered_map>

/*
 * The Runtime is the owners of all the system resources
 * It coordinates data handling and creating schedulers
 */
class Runtime {
  public:
    // Should return a future of the final result?
    std::shared_ptr<BaseDataHandle> commit_graph(const TaskGraph &task_graph);

  private:
    std::shared_ptr<ThreadPool> thread_pool;
    std::shared_ptr<IGPUExecutor> gpu_exec;
};

#endif
