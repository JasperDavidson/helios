#ifndef SCHEDULER_H
#define SCHEDULER_H

#include "DataManager.h"
#include "IGPUExecutor.h"
#include "Tasks.h"
#include "ThreadPool.h"
#include <memory>

class Scheduler {
  public:
    Scheduler(const DataManager &data_manager, std::shared_ptr<ThreadPool> thread_pool,
              std::shared_ptr<IGPUExecutor> gpu_executor)
        : data_manager(data_manager), thread_pool(thread_pool), gpu_executor(gpu_executor) {};

    // TODO: For both visit methods, implement event polling -> wrap in a lambda that triggers queue cond. var.
    template <typename F, class... Types> void visit(const CPUTask<F, Types...> &cpu_task) const {
        thread_pool->add_task(cpu_task.task_lambda);
    };
    void visit(const GPUTask &gpu_task) const;

    bool check_kernel_status(const std::string &kernel_name) { return gpu_executor->get_kernel_status(kernel_name); }

    template <typename F, class... Types> void execute_graph(const TaskGraph &task_graph);

  private:
    DataManager data_manager;
    std::shared_ptr<ThreadPool> thread_pool;
    std::shared_ptr<IGPUExecutor> gpu_executor;
};

#endif
